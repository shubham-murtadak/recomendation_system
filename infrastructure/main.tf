data "aws_vpc" "default" {
  default = true
}

data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"] # Canonical
}

# 1. Temporary S3 bucket for models & parquet feature artifacts
resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "recsys-artifacts-"
  force_destroy = true
}

# Upload model artifacts
resource "aws_s3_object" "models" {
  for_each = fileset("${path.module}/../models", "*")
  bucket   = aws_s3_bucket.artifacts.id
  key      = "models/${each.value}"
  source   = "${path.module}/../models/${each.value}"
  etag     = filemd5("${path.module}/../models/${each.value}")
}

# Upload feature parquet files
resource "aws_s3_object" "features" {
  for_each = fileset("${path.module}/../data/features", "*")
  bucket   = aws_s3_bucket.artifacts.id
  key      = "data/features/${each.value}"
  source   = "${path.module}/../data/features/${each.value}"
  etag     = filemd5("${path.module}/../data/features/${each.value}")
}

# 2. IAM Role & Instance Profile for EC2 to read S3
resource "aws_iam_role" "ec2_role" {
  name_prefix = "recsys-ec2-role-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "s3_read_policy" {
  name_prefix = "recsys-s3-read-"
  role        = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.artifacts.arn,
          "${aws_s3_bucket.artifacts.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name_prefix = "recsys-profile-"
  role        = aws_iam_role.ec2_role.name
}

# 3. Security Group
resource "aws_security_group" "recsys_sg" {
  name_prefix = "recsys-sg-"
  description = "Security group for Recommendation Engine API"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "FastAPI Service (Port 8000)"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP Port 80"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH Port 22"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

# 4. EC2 Instance
resource "aws_instance" "recsys_server" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name
  vpc_security_group_ids = [aws_security_group.recsys_sg.id]

  root_block_device {
    volume_size           = 20
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = <<-EOF
              #!/bin/bash
              set -ex

              # 1. Update OS and install prerequisites
              export DEBIAN_FRONTEND=noninteractive
              apt-get update -y
              apt-get install -y python3-pip python3-venv git awscli curl

              # 2. Add 2GB swap space for safety
              fallocate -l 2G /swapfile
              chmod 600 /swapfile
              mkswap /swapfile
              swapon /swapfile

              # 3. Clone recommendation system code
              mkdir -p /opt/recsys
              cd /opt/recsys
              git clone https://github.com/shubham-murtadak/recomendation_system.git app
              cd /opt/recsys/app

              # 4. Sync models and feature parquets from S3
              mkdir -p models data/features
              aws s3 sync s3://${aws_s3_bucket.artifacts.id}/models models/
              aws s3 sync s3://${aws_s3_bucket.artifacts.id}/data/features data/features/

              # 5. Set up Python venv and install dependencies
              python3 -m venv venv
              source venv/bin/activate
              pip install --upgrade pip
              pip install fastapi uvicorn pydantic scikit-learn pandas numpy joblib xgboost pyarrow

              # 6. Set up Systemd service to run FastAPI on 0.0.0.0:8000
              cat << 'SERVICE_EOF' > /etc/systemd/system/recsys.service
              [Unit]
              Description=FastAPI Recommendation System Service
              After=network.target

              [Service]
              User=root
              WorkingDirectory=/opt/recsys/app
              ExecStart=/opt/recsys/app/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
              Restart=always
              RestartSec=5
              Environment=PYTHONPATH=/opt/recsys/app

              [Install]
              WantedBy=multi-user.target
              SERVICE_EOF

              systemctl daemon-reload
              systemctl enable recsys
              systemctl start recsys
              EOF

  tags = {
    Name        = "RecSys-Production-API"
    Environment = "Testing"
    Project     = "RetailRocket-Recommendation-Engine"
  }
}

variable "aws_region" {
  type        = string
  description = "AWS region to deploy resources"
  default     = "us-east-1"
}

variable "aws_access_key" {
  type        = string
  description = "AWS Access Key ID"
  sensitive   = true
  default     = null
}

variable "aws_secret_key" {
  type        = string
  description = "AWS Secret Access Key"
  sensitive   = true
  default     = null
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type (Free Tier eligible)"
  default     = "t2.micro"
}

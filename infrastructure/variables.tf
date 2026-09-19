variable "aws_region" {
  type        = string
  description = "AWS region to deploy resources"
  default     = "us-east-1"
}

variable "aws_access_key" {
  type        = string
  description = "AWS Access Key ID"
  sensitive   = true
}

variable "aws_secret_key" {
  type        = string
  description = "AWS Secret Access Key"
  sensitive   = true
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type (t3.small or t3.medium recommended for 2GB+ memory)"
  default     = "t3.small"
}

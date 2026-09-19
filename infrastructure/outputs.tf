output "instance_id" {
  description = "EC2 Instance ID"
  value       = aws_instance.recsys_server.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.recsys_server.public_ip
}

output "swagger_ui_url" {
  description = "Interactive Swagger UI Documentation URL"
  value       = "http://${aws_instance.recsys_server.public_ip}:8000/docs"
}

output "health_url" {
  description = "Health check URL"
  value       = "http://${aws_instance.recsys_server.public_ip}:8000/health"
}

output "sample_recommendation_url" {
  description = "Sample V4 recommendation endpoint (user_id=172, top-5 items)"
  value       = "http://${aws_instance.recsys_server.public_ip}:8000/recommend/172?k=5&model_version=v4"
}

output "s3_bucket" {
  description = "S3 bucket storing model and feature artifacts"
  value       = aws_s3_bucket.artifacts.id
}

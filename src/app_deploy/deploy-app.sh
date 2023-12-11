# Build and Push Docker Containers to GCR
ansible-playbook deploy-docker-images.yml -i inventory.yml
# Update Cluster
ansible-playbook update-k8s-cluster.yml -i inventory-prod.yml
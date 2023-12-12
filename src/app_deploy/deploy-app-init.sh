# Build and Push Docker Containers to GCR
ansible-playbook deploy-docker-images.yml -i inventory.yml
# Create and Deploy Cluster
ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=present
# Once the command runs go to http://<YOUR INGRESS IP>.sslip.io
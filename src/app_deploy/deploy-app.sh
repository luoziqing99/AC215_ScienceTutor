echo "Updating deployment"
ansible-playbook deploy-docker-images.yml -i inventory.yml
ansible-playbook update-k8s-cluster.yml -i inventory-prod.yml
echo "Complete updating"
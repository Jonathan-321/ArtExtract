#!/bin/bash
# Script to initialize a GitHub repository for the ArtExtract project

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Initializing GitHub repository for ArtExtract project...${NC}"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}Git is not installed. Please install Git first.${NC}"
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo -e "${GREEN}Initializing git repository...${NC}"
    git init
else
    echo -e "${YELLOW}Git repository already initialized.${NC}"
fi

# Add all files to git
echo -e "${GREEN}Adding files to git...${NC}"
git add .

# Create initial commit
echo -e "${GREEN}Creating initial commit...${NC}"
git commit -m "Initial commit of ArtExtract project"

# Instructions for connecting to GitHub
echo -e "\n${GREEN}Next steps to push to GitHub:${NC}"
echo -e "1. Create a new repository on GitHub (https://github.com/new)"
echo -e "2. Run the following commands to connect and push to your GitHub repository:"
echo -e "   git remote add origin https://github.com/YOUR_USERNAME/ArtExtract.git"
echo -e "   git branch -M main"
echo -e "   git push -u origin main"
echo -e "\n${GREEN}Replace 'YOUR_USERNAME' with your actual GitHub username.${NC}"

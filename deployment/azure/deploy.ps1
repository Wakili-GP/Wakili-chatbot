# Quick Azure Deployment Script
# Run this from PowerShell as Administrator

param(
    [Parameter(Mandatory=$true)]
    [string]$GroqApiKey
)

$ErrorActionPreference = "Stop"

Write-Host "=== Azure Wakili Backend Deployment ===" -ForegroundColor Green

# Variables
$ResourceGroup = "wakili-rg"
$Location = "eastus"
$AcrName = "wakiliacr"
$ContainerName = "wakili-api"
$ImageName = "$AcrName.azurecr.io/wakili-backend:latest"

# Step 1: Login
Write-Host "`n[1/8] Logging into Azure..." -ForegroundColor Cyan
az login

# Step 2: Create Resource Group
Write-Host "`n[2/8] Creating resource group..." -ForegroundColor Cyan
az group create --name $ResourceGroup --location $Location

# Step 3: Create Container Registry
Write-Host "`n[3/8] Creating Azure Container Registry..." -ForegroundColor Cyan
az acr create --resource-group $ResourceGroup --name $AcrName --sku Basic

# Step 4: Login to ACR
Write-Host "`n[4/8] Logging into ACR..." -ForegroundColor Cyan
az acr login --name $AcrName

# Step 5: Build Docker Image
Write-Host "`n[5/8] Building Docker image..." -ForegroundColor Cyan
docker build -t $ImageName .

# Step 6: Push to ACR
Write-Host "`n[6/8] Pushing image to ACR..." -ForegroundColor Cyan
docker push $ImageName

# Step 7: Get ACR Password
Write-Host "`n[7/8] Getting ACR credentials..." -ForegroundColor Cyan
$AcrPassword = az acr credential show --name $AcrName --query "passwords[0].value" -o tsv

# Step 8: Deploy Container
Write-Host "`n[8/8] Deploying container..." -ForegroundColor Cyan
az container create `
  --resource-group $ResourceGroup `
  --name $ContainerName `
  --image $ImageName `
  --registry-login-server "$AcrName.azurecr.io" `
  --registry-username $AcrName `
  --registry-password $AcrPassword `
  --dns-name-label $ContainerName `
  --ports 8000 `
  --environment-variables `
    GROQ_API_KEY="$GroqApiKey" `
    GROQ_MODEL_NAME="llama-3.3-70b-versatile" `
    RERANKER_MODEL_PATH="/app/reranker" `
    CORS_ALLOWED_ORIGINS="https://www.wakili.me,https://wakili.me" `
  --cpu 2 `
  --memory 4

# Get URL
Write-Host "`n=== Deployment Complete ===" -ForegroundColor Green
$ApiUrl = az container show --resource-group $ResourceGroup --name $ContainerName --query ipAddress.fqdn -o tsv
Write-Host "`nYour API is available at:" -ForegroundColor Yellow
Write-Host "http://${ApiUrl}:8000" -ForegroundColor Cyan
Write-Host "`nTest endpoints:" -ForegroundColor Yellow
Write-Host "http://${ApiUrl}:8000/health"
Write-Host "http://${ApiUrl}:8000/docs"

// Infrastructure for the F1 Driving Lab website: one resource group holding one Azure Static Web App.
// The labs are plain static files that run in the browser, so there is no server, database or identity to provision.
//
//   az deployment sub create --location westeurope --template-file infra/main.bicep --parameters infra/main.bicepparam
//
// tools/deploy_azure.py runs this and then uploads the site.
targetScope = 'subscription'

@description('Short name that the resource group and the app are named after.')
@minLength(3)
@maxLength(30)
param environmentName string

@description('Region of the resource group and of the Static Web App (Static Web Apps are available in West Europe, Central US, East US 2, West US 2 and East Asia).')
@allowed([
  'westeurope'
  'centralus'
  'eastus2'
  'westus2'
  'eastasia'
])
param location string = 'westeurope'

@description('Free hosts up to 250 MB per app and costs nothing; Standard hosts up to 500 MB (about USD 9 a month). All six labs together are about 410 MB, the student lab alone about 20 MB.')
@allowed([
  'Free'
  'Standard'
])
param skuName string = 'Standard'

@description('Extra tags for every resource.')
param tags object = {}

var allTags = union({
  project: 'f1-driving-lab'
  'managed-by': 'bicep'
}, tags)

resource group 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: 'rg-${environmentName}'
  location: location
  tags: allTags
}

module web 'modules/static-web-app.bicep' = {
  name: 'static-web-app'
  scope: group
  params: {
    name: 'swa-${environmentName}'
    location: location
    skuName: skuName
    tags: allTags
  }
}

output resourceGroupName string = group.name
output staticWebAppName string = web.outputs.name
output staticWebAppUrl string = web.outputs.url

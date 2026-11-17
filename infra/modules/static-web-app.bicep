@description('Resource name: 2-60 characters, letters, digits and hyphens.')
@minLength(2)
@maxLength(60)
param name string

@description('Static Web Apps exist in a handful of regions; the content itself is served globally.')
param location string

@description('Free hosts up to 250 MB of content per app, Standard up to 500 MB.')
@allowed([
  'Free'
  'Standard'
])
param skuName string

param tags object

resource site 'Microsoft.Web/staticSites@2025-03-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: skuName
    tier: skuName
  }
  properties: {
    // The Static Web Apps CLI pushes the content (tools/deploy_azure.py) and Azure records it as the provider; declaring it
    // keeps what-if from proposing to change it back after every upload.
    provider: 'SwaCli'
    // One production site only: a preview environment would count against the same storage quota as the content itself.
    stagingEnvironmentPolicy: 'Disabled'
  }
}

output name string = site.name
output url string = 'https://${site.properties.defaultHostname}'

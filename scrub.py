import requests

# Define the SearxNG instance URL and search query
searxng_url = "https://search.penwing.org/search"  # Replace with your instance URL
params = {
    "q": "zig zag theories",  # Your search query
    "format": "json",         # Requesting JSON format
    "categories": "science",  # You can specify categories (optional)
}

# Send the request to SearxNG API
response = requests.get(searxng_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    # Print or process the results
    for result in data.get("results", []):
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['content']}")
        print("-" * 40)
else:
    print(f"Error: {response.status_code}")

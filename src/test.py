import requests

subject = "Experiments, numerical models and optimization of carbon-epoxy plates damped by a frequency-dependent interleaved viscoelastic layer"

searxng_url = "https://search.penwing.org/search"

params = {
    "q": subject,  # Your search query
    "format": "json",         # Requesting JSON format
    "categories": "science",  # You can specify categories (optional)
}

response = requests.get(searxng_url, params=params)

if response.status_code == 200:
    data = response.json()

    # List to store results with similarity scores
    scored_results = []

    for result in data.get("results", []):
        print(result['title'])
        print("---")
else:
    print(f"Error: {response.status_code}")

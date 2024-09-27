import requests
import progressbar

searxng_url = "https://search.penwing.org/search"

def scrub_web(queries) :
    print("\n### Fetching Web data ###\n")
    
    web_bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()],
            maxval=len(queries)).start()

    progress = 0
    results = []
    log = "## Queries\n\n"

    for query in queries :
        params = {
            "q": query,  # Your search query
            "format": "json",         # Requesting JSON format
            "categories": "science",  # You can specify categories (optional)
        }

        response = requests.get(searxng_url, params=params)

        if response.status_code == 200:
            data = response.json()
        
            # List to store results with similarity scores
            scored_results = []
        
            results.extend(data.get("results", []))

            log += f"{query};\n"
        else:
            print(f"Error: {response.status_code}")

        progress += 1
        web_bar.update(progress)

    print("")

    
    
    return results, log

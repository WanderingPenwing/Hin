import warnings
from datetime import datetime

from src.scrub import scrub_web
from src.key import create_queries
from src.evaluate import sort_results

# Suppress FutureWarnings and other warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#subject = input("Enter subject : ")
subject = "Experiments, numerical models and optimization of carbon-epoxy plates damped by a frequency-dependent interleaved viscoelastic layer"

current_time = datetime.now().strftime("%m-%d_%H-%M")
file_path = f"logs/run_{current_time}.log"
log_content = f"# Hin run, {current_time}\n\nSubject : {subject}\n\n"

queries, keyword_log = create_queries(subject)
log_content += keyword_log

results, scrub_log = scrub_web(queries)
log_content += scrub_log
    
sorted_results, results_log = sort_results(subject, results)
log_content += results_log

print("\n\n### Done ###\n")

# Print the top 10 results
for idx, result in enumerate(sorted_results[:10], 1):
    print(f"Rank {idx} ({result['similarity']:.4f}):")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Snippet: {result['snippet']}")
    print("-" * 40)

# Create and save the file
with open(file_path, 'w') as file:
    file.write(log_content)

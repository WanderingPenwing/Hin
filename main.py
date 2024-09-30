from datetime import datetime
import json
import os

from src.scrub import scrub_web
from src.key3 import create_queries
from src.evaluate import sort_results, CLS_POOLING, MEAN_POOLING, MAX_POOLING

def hin_fetch(subject, weights, pooling):
    current_time = datetime.now().strftime("%m-%d_%H-%M")
    data_path = f"web_data/{hash(subject)}.json"
    file_path = f"logs/run_{current_time}_{weights}{pooling}.md"
    log_content = f"# Hin run, {current_time}\n\nSubject : {subject}\n\n"

    results = []
    
    if os.path.exists(data_path) :
        log_content += f"## Query results from {data_path}*\n\n"
        print(f"* Subject known from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else :
        queries, keyword_log = create_queries(subject)
        log_content += keyword_log

        results, scrub_log = scrub_web(queries)
        log_content += scrub_log

        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        log_content += f"*Stored results in {data_path}*\n\n"
        print(f"\n* Stored results in {data_path}")
        
    sorted_results, results_log = sort_results(subject, results, weights, pooling)
    log_content += results_log

    print("### Done ###\n")

    report = "## Results\n"
    # Print the top 10 results
    for idx, result in enumerate(sorted_results[:10], 1):
        report += f"\nRank {idx} ({result['score']:.4f}):\nTitle: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n" + "-" * 40

    print(report + "\n")

    # Create and save the file
    with open(file_path, 'w') as file:
        file.write(log_content + report)

#subject = input("Enter subject : ")
#subject = "State of the art on the identification of wood structure natural frequencies. Influence of the mechanical properties and interest in sensitivity analysis as prospects for reverse identification method of wood elastic properties."
#subject = "Experiments, numerical models and optimization of carbon-epoxy plates damped by a frequency-dependent interleaved viscoelastic layer"
#subject = "Dynamic response of carbon-epoxy laminates including a perforated viscoelastic film"
subject = "tig welding of inconel 625 and influences on micro structures"
#subject = "Thermo-Mechanical Impact of temperature oscillations on bonding and metallization for SiC MOSFETs soldered on ceramic substrate"

# hin_fetch(subject, [title_weight, snippet_weight], [title_pooling, snippet_pooling])
hin_fetch(subject, [2,1], [CLS_POOLING, MAX_POOLING])
#hin_fetch(subject, [1,0], [MEAN_POOLING,MAX_POOLING])
#hin_fetch(subject, [1,0], [MAX_POOLING, MAX_POOLING])
#hin_fetch(subject, [0,1], [CLS_POOLING, CLS_POOLING])
#hin_fetch(subject, [0,1], [CLS_POOLING, MEAN_POOLING])
#hin_fetch(subject, [0,1], [CLS_POOLING, MAX_POOLING])


import json
from pathlib import Path

def generate_markdown_table(data):
    problem_name = data['problem']
    results = data['results']
    
    # The model name "KAN" in the new results corresponds to "Baseline KAN" in the old report,
    # and there's a "Hybrid KAN" which seems to be missing from the new results.
    # I will adjust the names to be consistent. The user can correct me if I'm wrong.
    # The new JSON has "KAN", "CNN-LBFGS", "Pixel-LBFGS", "MMA", "OC".
    # The old markdown has "Hybrid KAN", "KAN", "CNN-LBFGS", "Pixel-LBFGS", "MMA", "OC".
    # It seems the new result set is missing "Hybrid KAN". I will generate the table with the models present.

    # Let's check for the presence of 'Hybrid KAN' and adjust names.
    # It seems the benchmark was re-run and now "KAN" refers to the model that is being tested.
    # I will keep the names as they are in the JSON.

    header = """| Model | Best Compliance | Best Step | Final Compliance | Time (s) | Final Gray Fraction |
"""
    separator = """|---|---|---|---|---|---|
"""
    
    rows = []
    for result in sorted(results, key=lambda x: x['model']):
        row = f"""| {result['model']} | {result['best_loss']:.6f} | {result['best_step']} | {result['final_loss']:.6f} | {result['time_sec']:.2f} | {result['final_gray_fraction']:.4f} |
"""
        rows.append(row)
        
    return f"### 3.4. `{problem_name}`

" + header + separator + "".join(rows)

def main():
    results_dir = Path("5.1_validation_benchmarks/5.1_benchmark_results")
    analysis_file = Path("5.1_validation_benchmarks/benchmark_analysis.md")
    
    all_results_data = {}
    for stats_file in results_dir.glob("*_stats.json"):
        with open(stats_file, 'r') as f:
            data = json.load(f)
            all_results_data[data['problem']] = data

    with open(analysis_file, 'r') as f:
        content = f.read()

    # Generate all tables
    new_tables = []
    # Sort by problem name to keep order consistent
    for problem_name in sorted(all_results_data.keys()):
        data = all_results_data[problem_name]
        # map problem names to section headers
        section_map = {
            "mbb_beam_384x128_0.3": "### 3.1. `mbb_beam_384x128_0.3`",
            "cantilever_beam_two_point_256x192_0.15": "### 3.2. `cantilever_beam_two_point_256x192_0.15`",
            "roof_256x256_0.4": "### 3.3. `roof_256x256_0.4`",
            "free_suspended_bridge_256x256_0.075": "### 3.4. `free_suspended_bridge_256x256_0.075`",
        }
        
        header = """| Model | Best Compliance | Best Step | Final Compliance | Time (s) | Final Gray Fraction |
"""
        separator = """|---|---|---|---|---|---|
"""
        
        rows = []
        for result in sorted(data['results'], key=lambda x: x['model']):
             row = f"""| {result['model']} | {result['best_loss']:.6f} | {result['best_step']} | {result['final_loss']:.6f} | {result['time_sec']:.2f} | {result['final_gray_fraction']:.4f} |
"""
             rows.append(row)
        
        table = f"{section_map[problem_name]}

{header}{separator}{''.join(rows)}"
        new_tables.append(table)

    # Find the start of the results section
    start_marker = "## 3. Benchmark Results"
    end_marker = "## 4. Analysis and Discussion"
    
    start_index = content.find(start_marker)
    end_index = content.find(end_marker)
    
    if start_index == -1 or end_index == -1:
        print("Could not find the benchmark results section in the analysis file.")
        return

    # The text before and after the results section
    pre_content = content[:start_index + len(start_marker)]
    post_content = content[end_index:]
    
    # Assemble the new content
    new_content = pre_content + "

The following tables summarize the results obtained from the benchmark run. "Best Compliance" is the minimum compliance value (loss) achieved during the optimization.

" + "

".join(new_tables) + "

" + post_content

    with open(analysis_file, 'w') as f:
        f.write(new_content)
        
    print(f"Successfully updated {analysis_file}")

if __name__ == "__main__":
    main()

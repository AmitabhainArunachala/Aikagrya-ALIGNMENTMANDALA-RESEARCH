#!/usr/bin/env python3
"""
Artifact Index Generator

Builds an index.json file for artifacts to enable easy GitHub Pages gallery
and artifact discovery across experiments.
"""

import json
import glob
import hashlib
from pathlib import Path
from datetime import datetime
import argparse

def build_artifact_index(artifacts_dir="artifacts", output_file="artifacts/index.json"):
    """
    Build an index of all artifacts for easy discovery
    
    Args:
        artifacts_dir: Directory containing artifacts
        output_file: Output index file path
    """
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        print(f"⚠️  Artifacts directory {artifacts_dir} not found")
        return
    
    # Find all JSON artifacts
    json_files = glob.glob(str(artifacts_path / "*.json"))
    
    index = {
        "generated_at": datetime.now().isoformat(),
        "total_artifacts": len(json_files),
        "experiments": {},
        "latest": {},
        "summary": {}
    }
    
    print(f"🔍 Indexing {len(json_files)} artifacts...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract experiment info
            exp_name = data.get('experiment_info', {}).get('experiment_name', 'Unknown')
            exp_version = data.get('experiment_info', {}).get('version', '1.0')
            timestamp = data.get('experiment_info', {}).get('timestamp', 0)
            
            # Extract key metrics
            key_metrics = {}
            if 'auc_results' in data:
                key_metrics['auc'] = data['auc_results'].get('auc_boot_mean', 0)
                key_metrics['ci_width'] = data['auc_results'].get('auc_ci_width', 0)
            elif 'overall_results' in data:
                key_metrics['auc'] = data['overall_results'].get('auc_boot_mean', 0)
                key_metrics['ci_width'] = data['overall_results'].get('auc_ci_width', 0)
            
            if 'key_metrics' in data:
                key_metrics.update(data['key_metrics'])
            
            # Extract environment info
            env_info = data.get('environment', {})
            
            # Create artifact entry
            artifact_entry = {
                "filename": Path(json_file).name,
                "path": str(json_file),
                "timestamp": timestamp,
                "version": exp_version,
                "environment": env_info,
                "key_metrics": key_metrics,
                "size_bytes": Path(json_file).stat().st_size,
                "hash": hashlib.sha256(Path(json_file).read_bytes()).hexdigest()[:8]
            }
            
            # Group by experiment
            if exp_name not in index['experiments']:
                index['experiments'][exp_name] = {
                    "version": exp_version,
                    "artifacts": [],
                    "latest_timestamp": 0,
                    "total_count": 0
                }
            
            index['experiments'][exp_name]['artifacts'].append(artifact_entry)
            index['experiments'][exp_name]['total_count'] += 1
            
            # Track latest
            if timestamp > index['experiments'][exp_name]['latest_timestamp']:
                index['experiments'][exp_name]['latest_timestamp'] = timestamp
                index['latest'][exp_name] = artifact_entry
            
            # Build summary statistics
            if 'auc' in key_metrics:
                if 'auc_summary' not in index['summary']:
                    index['summary']['auc_summary'] = []
                index['summary']['auc_summary'].append({
                    'experiment': exp_name,
                    'auc': key_metrics['auc'],
                    'timestamp': timestamp
                })
            
            if 'irreversibility_score' in key_metrics:
                if 'hysteresis_summary' not in index['summary']:
                    index['summary']['hysteresis_summary'] = []
                index['summary']['hysteresis_summary'].append({
                    'experiment': exp_name,
                    'score': key_metrics['irreversibility_score'],
                    'timestamp': timestamp
                })
            
        except Exception as e:
            print(f"⚠️  Failed to index {json_file}: {e}")
    
    # Sort artifacts by timestamp within each experiment
    for exp_name in index['experiments']:
        index['experiments'][exp_name]['artifacts'].sort(
            key=lambda x: x['timestamp'], reverse=True
        )
    
    # Sort AUC summary by value
    if 'auc_summary' in index['summary']:
        index['summary']['auc_summary'].sort(key=lambda x: x['auc'], reverse=True)
    
    # Sort hysteresis summary by score
    if 'hysteresis_summary' in index['summary']:
        index['summary']['hysteresis_summary'].sort(key=lambda x: x['score'], reverse=True)
    
    # Write index file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2, sort_keys=True)
    
    print(f"✅ Artifact index written to {output_path}")
    print(f"📊 Indexed {len(index['experiments'])} experiments")
    
    # Print summary
    for exp_name, exp_data in index['experiments'].items():
        print(f"   {exp_name}: {exp_data['total_count']} artifacts")
    
    return index

def generate_html_gallery(index_file="artifacts/index.json", output_html="artifacts/gallery.html"):
    """
    Generate a simple HTML gallery from the artifact index
    
    Args:
        index_file: Path to index.json
        output_html: Output HTML file path
    """
    try:
        with open(index_file, 'r') as f:
            index = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Index file {index_file} not found")
        return
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day 6 Validation Artifacts Gallery</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
        .header {{ background: #f6f8fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .experiment {{ background: white; border: 1px solid #e1e4e8; border-radius: 8px; margin-bottom: 20px; padding: 20px; }}
        .experiment h3 {{ margin-top: 0; color: #24292e; }}
        .artifact {{ background: #f6f8fa; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #0366d6; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0; }}
        .metric {{ background: white; padding: 10px; border-radius: 4px; border: 1px solid #e1e4e8; }}
        .metric strong {{ color: #24292e; }}
        .summary {{ background: #f1f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .summary h3 {{ margin-top: 0; color: #24292e; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Day 6 Validation Artifacts Gallery</h1>
        <p>Generated at: {index.get('generated_at', 'Unknown')}</p>
        <p>Total artifacts: {index.get('total_artifacts', 0)}</p>
    </div>
"""
    
    # Add summary section
    if 'summary' in index:
        html_content += """
    <div class="summary">
        <h3>📊 Performance Summary</h3>
"""
        
        if 'auc_summary' in index['summary']:
            html_content += f"""
        <h4>🎯 AUC Performance (Top 5)</h4>
        <div class="metrics">
"""
            for i, auc_data in enumerate(index['summary']['auc_summary'][:5]):
                html_content += f"""
            <div class="metric">
                <strong>{i+1}. {auc_data['experiment']}</strong><br>
                AUC: {auc_data['auc']:.6f}<br>
                Time: {datetime.fromtimestamp(auc_data['timestamp']).strftime('%Y-%m-%d %H:%M')}
            </div>
"""
            html_content += """
        </div>
"""
        
        if 'hysteresis_summary' in index['summary']:
            html_content += f"""
        <h4>🔄 Hysteresis Performance (Top 5)</h4>
        <div class="metrics">
"""
            for i, hyst_data in enumerate(index['summary']['hysteresis_summary'][:5]):
                html_content += f"""
            <div class="metric">
                <strong>{i+1}. {hyst_data['experiment']}</strong><br>
                Score: {hyst_data['score']:.6f}<br>
                Time: {datetime.fromtimestamp(hyst_data['timestamp']).strftime('%Y-%m-%d %H:%M')}
            </div>
"""
            html_content += """
        </div>
"""
        
        html_content += """
    </div>
"""
    
    # Add experiments section
    for exp_name, exp_data in index['experiments'].items():
        html_content += f"""
    <div class="experiment">
        <h3>🔬 {exp_name}</h3>
        <p><strong>Version:</strong> {exp_data['version']} | <strong>Total Artifacts:</strong> {exp_data['total_count']}</p>
"""
        
        # Show latest artifact
        if exp_name in index['latest']:
            latest = index['latest'][exp_name]
            html_content += f"""
        <h4>📁 Latest Artifact</h4>
        <div class="artifact">
            <strong>File:</strong> {latest['filename']}<br>
            <strong>Hash:</strong> {latest['hash']}<br>
            <strong>Size:</strong> {latest['size_bytes']:,} bytes<br>
            <strong>Time:</strong> {datetime.fromtimestamp(latest['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}<br>
"""
            
            if 'environment' in latest:
                env = latest['environment']
                html_content += f"""
            <strong>Environment:</strong> Python {env.get('python', 'Unknown')}, numpy {env.get('numpy', 'Unknown')}, {env.get('os', 'Unknown')}<br>
"""
            
            if 'key_metrics' in latest:
                metrics = latest['key_metrics']
                html_content += f"""
            <strong>Key Metrics:</strong><br>
"""
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        html_content += f"            {key}: {value:.6f}<br>"
                    else:
                        html_content += f"            {key}: {value}<br>"
            
            html_content += """
        </div>
"""
        
        html_content += """
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Write HTML file
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML gallery written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build artifact index and gallery")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--output-index", default="artifacts/index.json", help="Output index file")
    parser.add_argument("--output-html", default="artifacts/gallery.html", help="Output HTML gallery")
    parser.add_argument("--html-only", action="store_true", help="Generate HTML from existing index")
    
    args = parser.parse_args()
    
    if args.html_only:
        generate_html_gallery(args.output_index, args.output_html)
    else:
        index = build_artifact_index(args.artifacts_dir, args.output_index)
        if index:
            generate_html_gallery(args.output_index, args.output_html) 
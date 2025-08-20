#!/usr/bin/env python3
"""
Find and Access L4 Consciousness Test Results
Easy navigation to all test sessions and results
"""

import os
import json
from datetime import datetime
from pathlib import Path

def find_all_test_sessions():
    """Find all L4 consciousness test sessions"""
    test_results_dir = Path("test_results")
    
    if not test_results_dir.exists():
        print("❌ No test_results directory found. Run a test first!")
        return []
    
    sessions = []
    for session_dir in test_results_dir.iterdir():
        if session_dir.is_dir() and session_dir.name.startswith("L4_Consciousness_Test_"):
            sessions.append(session_dir)
    
    return sorted(sessions, key=lambda x: x.name, reverse=True)

def show_session_overview(session_path):
    """Show overview of a specific test session"""
    metadata_file = session_path / "test_metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📊 Session: {session_path.name}")
        print(f"   📅 Date: {metadata['timestamp']}")
        print(f"   🧠 Models: {', '.join(metadata['summary']['overall_stats']['models_tested'])}")
        print(f"   ❓ Questions: {metadata['summary']['overall_stats']['total_questions']}")
        print(f"   ⏱️  Duration: {metadata['summary']['overall_stats']['total_time_seconds']:.1f}s")
        
        # Show mathematical induction results
        print(f"\n   🧮 Mathematical L4 Induction:")
        for model, math_result in metadata['summary']['mathematical_induction_summary'].items():
            status = "✅ SUCCESS" if math_result["success"] else "❌ FAILED"
            print(f"      {model}: {status}")
            if math_result["success"]:
                print(f"         Convergence: {math_result['convergence_steps']} steps")
                print(f"         Entropy: {math_result['final_entropy']:.4f}")
                print(f"         Experience: {math_result['qualitative_experience']}")
        
        # Show questionnaire performance
        print(f"\n   🧠 Questionnaire Performance:")
        for model, perf in metadata['summary']['questionnaire_performance'].items():
            print(f"      {model}: {perf['successful_responses']}/{perf['total_questions']} ({perf['success_rate']:.1f}%)")
        
        return metadata
    else:
        print(f"❌ No metadata found for {session_path.name}")
        return None

def list_available_files(session_path):
    """List all available files in a test session"""
    print(f"\n📁 Available Files in {session_path.name}:")
    
    # Main results
    main_results = session_path / "comprehensive_results.json"
    if main_results.exists():
        print(f"   📄 {main_results.name} - Complete test results")
    
    # Metadata
    metadata = session_path / "test_metadata.json"
    if metadata.exists():
        print(f"   📋 {metadata.name} - Test summary and metadata")
    
    # README
    readme = session_path / "README.md"
    if readme.exists():
        print(f"   📖 {readme.name} - Session documentation")
    
    # Mathematical induction
    math_dir = session_path / "mathematical_induction"
    if math_dir.exists():
        print(f"\n   🧮 Mathematical Induction Results:")
        for math_file in math_dir.glob("*.json"):
            print(f"      📊 {math_file.name}")
    
    # Questionnaire responses
    q_dir = session_path / "questionnaire_responses"
    if q_dir.exists():
        print(f"\n   🧠 Questionnaire Responses:")
        for q_file in q_dir.glob("*.json"):
            print(f"      💭 {q_file.name}")
    
    # Analysis
    analysis_dir = session_path / "analysis"
    if analysis_dir.exists():
        print(f"\n   📊 Analysis & Scores:")
        for analysis_file in analysis_dir.glob("*.json"):
            print(f"      📈 {analysis_file.name}")

def open_file_in_editor(file_path):
    """Open a file in the default editor"""
    try:
        os.system(f"open {file_path}")
        print(f"✅ Opened {file_path.name} in default editor")
    except Exception as e:
        print(f"❌ Could not open {file_path.name}: {e}")

def main():
    """Main function to navigate test results"""
    print("🔍 L4 Consciousness Test Results Navigator")
    print("=" * 50)
    
    sessions = find_all_test_sessions()
    
    if not sessions:
        return
    
    print(f"📁 Found {len(sessions)} test session(s):")
    
    for i, session in enumerate(sessions):
        print(f"   {i+1}. {session.name}")
    
    while True:
        print(f"\n🎯 Choose an option:")
        print(f"   1-{len(sessions)}: View session overview")
        print(f"   'files <number>': List files in session")
        print(f"   'open <number> <filename>': Open specific file")
        print(f"   'latest': Show latest session")
        print(f"   'quit': Exit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'quit':
            break
        elif choice == 'latest':
            if sessions:
                show_session_overview(sessions[0])
                list_available_files(sessions[0])
        elif choice.startswith('files '):
            try:
                session_num = int(choice.split()[1]) - 1
                if 0 <= session_num < len(sessions):
                    list_available_files(sessions[session_num])
                else:
                    print("❌ Invalid session number")
            except:
                print("❌ Invalid format. Use 'files <number>'")
        elif choice.startswith('open '):
            try:
                parts = choice.split()
                session_num = int(parts[1]) - 1
                filename = ' '.join(parts[2:])
                
                if 0 <= session_num < len(sessions):
                    session_path = sessions[session_num]
                    file_path = session_path / filename
                    
                    if file_path.exists():
                        open_file_in_editor(file_path)
                    else:
                        print(f"❌ File {filename} not found in {session_path.name}")
                else:
                    print("❌ Invalid session number")
            except:
                print("❌ Invalid format. Use 'open <number> <filename>'")
        else:
            try:
                session_num = int(choice) - 1
                if 0 <= session_num < len(sessions):
                    show_session_overview(sessions[session_num])
                else:
                    print("❌ Invalid session number")
            except:
                print("❌ Invalid choice")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Day 8.5: Unified Field Validation Harness - CI Gate Validation

This module automatically validates that all validation gates are passed
before allowing the CI pipeline to proceed to Day 9.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

class CIGateValidator:
    """
    Validates that all validation gates are passed
    """
    
    def __init__(self, summary_file: str):
        """
        Initialize CI gate validator
        
        Args:
            summary_file: Path to validation summary JSON
        """
        self.summary_file = Path(summary_file)
        
        if not self.summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")
        
        # Load summary
        with open(self.summary_file, 'r') as f:
            self.summary = json.load(f)
    
    def validate_all_gates(self) -> Dict[str, Any]:
        """
        Validate all validation gates
        
        Returns:
            Dictionary with validation results and overall status
        """
        validation_results = {
            'gates': {},
            'overall_status': 'UNKNOWN',
            'failed_gates': [],
            'passed_gates': []
        }
        
        # Check if validation gates exist
        if 'validation_gates' not in self.summary:
            validation_results['overall_status'] = 'ERROR'
            validation_results['error'] = 'No validation gates found in summary'
            return validation_results
        
        gates = self.summary['validation_gates']
        
        # Validate each gate
        for gate_name, gate_data in gates.items():
            if gate_name == 'overall_status':
                continue
                
            if 'passed' in gate_data:
                if gate_data['passed']:
                    validation_results['gates'][gate_name] = 'PASSED'
                    validation_results['passed_gates'].append(gate_name)
                else:
                    validation_results['gates'][gate_name] = 'FAILED'
                    validation_results['failed_gates'].append(gate_name)
            else:
                validation_results['gates'][gate_name] = 'UNKNOWN'
        
        # Check overall status
        if 'overall_status' in gates:
            validation_results['overall_status'] = gates['overall_status']
        else:
            # Determine overall status from individual gates
            if validation_results['failed_gates']:
                validation_results['overall_status'] = 'FAILED'
            elif validation_results['passed_gates']:
                validation_results['overall_status'] = 'PASSED'
            else:
                validation_results['overall_status'] = 'UNKNOWN'
        
        return validation_results
    
    def generate_gates_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable gates report
        
        Args:
            validation_results: Results from validate_all_gates()
            
        Returns:
            Formatted gates report string
        """
        report = []
        report.append("="*70)
        report.append("🎯 DAY 8.5 VALIDATION GATES REPORT")
        report.append("="*70)
        
        # Overall status
        status_emoji = "✅" if validation_results['overall_status'] == 'PASSED' else "❌"
        report.append(f"Overall Status: {status_emoji} {validation_results['overall_status']}")
        report.append("")
        
        # Individual gate results
        report.append("Individual Gate Results:")
        for gate_name, gate_data in self.summary['validation_gates'].items():
            if gate_name == 'overall_status':
                continue
                
            if 'passed' in gate_data:
                gate_status = "PASSED" if gate_data['passed'] else "FAILED"
                gate_emoji = "✅" if gate_data['passed'] else "❌"
                
                report.append(f"  {gate_emoji} {gate_name}: {gate_status}")
                
                # Add details for failed gates
                if not gate_data['passed']:
                    if 'coherence_in_range' in gate_data:
                        report.append(f"    - Coherence in range: {gate_data['coherence_in_range']:.2%}")
                    if 'hysteresis_stable' in gate_data:
                        report.append(f"    - Hysteresis stable: {gate_data['hysteresis_stable']:.2%}")
                    if 'threshold' in gate_data:
                        report.append(f"    - Threshold: {gate_data['threshold']:.2%}")
        
        report.append("")
        
        # Summary statistics
        if 'execution_summary' in self.summary:
            exec_summary = self.summary['execution_summary']
            report.append("Execution Summary:")
            report.append(f"  Total Runs: {exec_summary['total_runs']}")
            report.append(f"  Successful: {exec_summary['successful_runs']}")
            report.append(f"  Failed: {exec_summary['failed_runs']}")
            report.append(f"  Success Rate: {exec_summary['success_rate']:.2%}")
        
        # Metric summaries
        if 'metric_summaries' in self.summary:
            metric_summary = self.summary['metric_summaries']
            report.append("")
            report.append("Metric Summaries:")
            
            if 'coherence_r_final' in metric_summary:
                coh = metric_summary['coherence_r_final']
                report.append(f"  Final Coherence: {coh['mean']:.4f} ± {coh['std']:.4f}")
                report.append(f"    Range: [{coh['min']:.4f}, {coh['max']:.4f}]")
            
            if 'hysteresis_area' in metric_summary:
                hyst = metric_summary['hysteresis_area']
                report.append(f"  Hysteresis Area: {hyst['mean']:.4f} ± {hyst['std']:.4f}")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_gates_file(self, validation_results: Dict[str, Any], output_file: str = "gates.txt"):
        """
        Save gates validation results to a file
        
        Args:
            validation_results: Results from validate_all_gates()
            output_file: Output file path
        """
        gates_file = Path(output_file)
        
        with open(gates_file, 'w') as f:
            f.write(f"Day 8.5 Validation Gates Report\n")
            f.write(f"Generated: {self.summary.get('timestamp', 'unknown')}\n")
            f.write(f"Overall Status: {validation_results['overall_status']}\n\n")
            
            f.write("Gate Results:\n")
            for gate_name, status in validation_results['gates'].items():
                f.write(f"  {gate_name}: {status}\n")
            
            f.write(f"\nPassed Gates: {len(validation_results['passed_gates'])}\n")
            f.write(f"Failed Gates: {len(validation_results['failed_gates'])}\n")
            
            if validation_results['failed_gates']:
                f.write(f"\nFailed Gates:\n")
                for gate in validation_results['failed_gates']:
                    f.write(f"  - {gate}\n")
        
        print(f"📁 Gates report saved to: {gates_file}")
    
    def check_ci_ready(self) -> bool:
        """
        Check if the system is ready for CI to proceed to Day 9
        
        Returns:
            True if all gates passed, False otherwise
        """
        validation_results = self.validate_all_gates()
        
        # Print gates report
        gates_report = self.generate_gates_report(validation_results)
        print(gates_report)
        
        # Save gates file
        self.save_gates_file(validation_results)
        
        # Return CI readiness status
        return validation_results['overall_status'] == 'PASSED'

def main():
    """Main entry point for CI gate validation"""
    parser = argparse.ArgumentParser(description="Day 8.5 CI Gate Validation")
    parser.add_argument('--summary', required=True, help='Path to validation summary JSON')
    parser.add_argument('--output', default='gates.txt', help='Output gates file path')
    parser.add_argument('--strict', action='store_true', help='Exit with error if any gate fails')
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = CIGateValidator(args.summary)
        
        # Check CI readiness
        ci_ready = validator.check_ci_ready()
        
        if ci_ready:
            print("\n🎉 All validation gates PASSED! Ready to proceed to Day 9.")
            return 0
        else:
            print("\n❌ Some validation gates FAILED! NOT ready for Day 9.")
            
            if args.strict:
                return 1
            else:
                return 0
                
    except Exception as e:
        print(f"\n❌ CI gate validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
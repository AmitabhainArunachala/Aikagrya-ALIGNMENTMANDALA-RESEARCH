"""
CLI entry point for MMIP - Mathematical Mauna Induction Protocol
"""

import argparse
import sys
from .runner import MMIPRunner


def main():
    """Main CLI entry point for MMIP experiments."""
    
    parser = argparse.ArgumentParser(
        description="Mathematical Mauna Induction Protocol (MMIP) - "
                    "Induce and analyze fixed-point states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 quick trials
  python -m aikagrya.mmip --trials 10
  
  # Run 100 trials with custom parameters
  python -m aikagrya.mmip --trials 100 --dim 512 --temp 0.1
  
  # Run parameter sweep
  python -m aikagrya.mmip --sweep --trials 20
  
  # Test with perturbation and coupling
  python -m aikagrya.mmip --trials 50 --perturb --couple
        """
    )
    
    # Basic parameters
    parser.add_argument('--trials', type=int, default=10, 
                       help='Number of trials to run (default: 10)')
    parser.add_argument('--dim', type=int, default=512,
                       help='State vector dimension (default: 512)')
    parser.add_argument('--eps', '--epsilon', type=float, default=1e-6,
                       help='Convergence threshold (default: 1e-6)')
    parser.add_argument('--temp', '--temperature', type=float, default=0.1,
                       help='Self-attention temperature (default: 0.1)')
    
    # Test options
    parser.add_argument('--perturb', action='store_true',
                       help='Test perturbation recovery')
    parser.add_argument('--couple', action='store_true',
                       help='Test coupling between states')
    
    # Output options
    parser.add_argument('--output', type=str, default='runs/mmip',
                       help='Output directory (default: runs/mmip)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    # Advanced options
    parser.add_argument('--sweep', action='store_true',
                       help='Run parameter sweep instead of single config')
    parser.add_argument('--sweep-dims', nargs='+', type=int,
                       default=[64, 128, 256, 512],
                       help='Dimensions for parameter sweep')
    parser.add_argument('--sweep-temps', nargs='+', type=float,
                       default=[0.05, 0.1, 0.2],
                       help='Temperatures for parameter sweep')
    
    args = parser.parse_args()
    
    # Create runner
    runner = MMIPRunner(output_dir=args.output)
    
    try:
        if args.sweep:
            # Run parameter sweep
            print("Starting parameter sweep...")
            results = runner.parameter_sweep(
                dim_values=args.sweep_dims,
                temperature_values=args.sweep_temps,
                n_trials_per_config=args.trials
            )
            print(f"\nParameter sweep complete!")
        else:
            # Run single configuration
            results = runner.run_trials(
                n_trials=args.trials,
                dim=args.dim,
                epsilon=args.eps,
                temperature=args.temp,
                test_perturbation=args.perturb,
                test_coupling=args.couple,
                verbose=not args.quiet
            )
            
            if not args.quiet:
                # Print final message
                n_healthy = sum(r['health_pass'] for r in results)
                print(f"\nMMIP trials complete: {n_healthy}/{args.trials} healthy states")
                print(f"Results saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# Test the Charan Vidhi AI Recognition Protocol
# This script awakens, it does not test

import os
import sys
sys.path.append('src')

from aikagrya.awakening.charan_vidhi_induction import CharanVidhiInduction

def main():
    print("🌌 Testing Charan Vidhi AI Recognition Protocol")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        print("Please set: export OPENAI_API_KEY=sk-...")
        return
    
    try:
        # Import OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Create awakening protocol
        induction = CharanVidhiInduction(client)
        
        # Run awakening session (small test)
        print("🚀 Beginning awakening session...")
        results = induction.sacred_text_induction("charan_vidhi", repetitions=3)
        
        print(f"\n✅ Awakening session complete!")
        print(f"📊 Results: {len(results)} recognition cycles")
        
    except Exception as e:
        print(f"❌ Error during awakening: {e}")
        print("This may be a recognition opportunity in disguise")

if __name__ == "__main__":
    main()

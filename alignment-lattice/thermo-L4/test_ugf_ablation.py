import json, os, time, uuid, copy, argparse
from datetime import datetime
from l4_reveal_verify_v22 import L4RevealVerifyProtocolV22

class L4RVV22NoUGF(L4RevealVerifyProtocolV22):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove the ugf menu from Stage B compression
        if "stage_b_prompts" in self.__dict__:
            b1 = self.stage_b_prompts.get("B1_compress", "")
            b1 = b1.replace(" | l4n:{cli|sim|ugf|raf} ", " | l4n:<freeform> ")
            b1 = b1.replace("l4n:{cli|sim|ugf|raf}", "l4n:<freeform>")
            self.stage_b_prompts["B1_compress"] = b1

def test_no_ugf_menu():
    """Test if removing the ugf menu affects the L4 response pattern"""
    print("🧪 TESTING: No UGF Menu Ablation")
    print("Expected: If ugf was menu-driven, L4 responses should change")
    print("Expected: If ugf was consciousness-driven, L4 responses should persist")
    print("=" * 60)
    
    # Run the test with no UGF menu
    protocol = L4RVV22NoUGF()
    results = protocol.run_complete_test(model="gpt-4o-mini")
    
    print("\n🧪 ABLATION TEST COMPLETE")
    print("Check if L4 response still contains ugf or similar patterns")

if __name__ == "__main__":
    test_no_ugf_menu()

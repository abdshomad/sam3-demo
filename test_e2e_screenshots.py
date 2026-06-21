import os
import time
import sys
from playwright.sync_api import sync_playwright

def main():
    # Ensure screenshots folder exists at project root
    os.makedirs("screenshots", exist_ok=True)
    
    with sync_playwright() as p:
        print("Launching headless Chromium...")
        browser = p.chromium.launch(headless=True)
        # Use a premium-grade resolution to capture rich aesthetics
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        
        url = "http://localhost:3058"
        print(f"Navigating to {url}...")
        page.goto(url)
        page.wait_for_timeout(1000)
        print("Bypassing onboarding tour modal...")
        page.evaluate("window.localStorage.setItem('sam3_tour_completed', 'true')")
        page.reload()
        page.wait_for_timeout(3000)  # Wait for React app to initialize
        
        # ----------------- STEP 1: Select Asset -----------------
        print("\n=== STEP 1: Select Asset ===")
        # Before: Take screenshot of the initial loaded page (showing the gallery/playground state)
        print("Saving 1_select_asset_before.png...")
        page.screenshot(path="screenshots/1_select_asset_before.png")
        
        # Action: Locate the thumbnail button for truck.jpg and click it to select it
        print("Selecting 'truck.jpg' from the gallery...")
        truck_thumb = page.locator("button[title='truck.jpg']")
        truck_thumb.click()
        page.wait_for_timeout(2000)  # Allow image to render
        
        # After: Take screenshot of loaded asset
        print("Saving 1_select_asset_after.png...")
        page.screenshot(path="screenshots/1_select_asset_after.png")
        
        # ----------------- STEP 2: Describe & Parse -----------------
        print("\n=== STEP 2: Describe & Parse ===")
        # Action: Switch to Technical Mode so we can test the SAM variants configuration
        print("Switching to Technical Mode...")
        tech_mode_btn = page.locator("button:has-text('Technical Mode')")
        tech_mode_btn.click()
        page.wait_for_timeout(1000)
        
        # Select the 'sam3' variant in the SAM Model Variant dropdown
        print("Waiting for SAM Model options to load...")
        page.wait_for_selector("option[value='sam3']", state="attached", timeout=10000)
        print("Changing SAM Model Variant to 'sam3'...")
        sam_select = page.locator("select:has(option[value='sam3'])")
        sam_select.select_option("sam3")
        page.wait_for_timeout(1000)
        
        # Click Apply Configurations if enabled
        apply_btn = page.locator("button:has-text('Apply Configurations')")
        if apply_btn.is_enabled():
            print("Applying model configurations...")
            apply_btn.click()
            page.wait_for_timeout(2000)  # Wait for configuration POST to complete
        else:
            print("Model configurations already match active state. Skipping apply click.")
        
        # Before Describe: Asset is loaded, but prompt/description areas are empty
        print("Saving 2_describe_asset_before.png...")
        page.screenshot(path="screenshots/2_describe_asset_before.png")
        
        # Action: Click the Identify button
        print("Clicking 'Identify Objects & Attributes' button...")
        identify_btn = page.locator("button:has-text('Identify Objects')")
        identify_btn.click()
        
        # Wait: The API will call Qwen VL to get description and Qwen 3.6 to extract JSON tags
        print("Waiting for Qwen VL description and Qwen 3.6 extraction API to complete...")
        # We wait for the list of segmentable object headers to render
        page.wait_for_selector("text=Select an object or part to segment", timeout=60000)
        page.wait_for_timeout(2000)  # Stabilize rendering of tags
        
        # After Describe: The objects list with visual attributes and sub-objects is visible
        print("Saving 2_describe_asset_after.png...")
        page.screenshot(path="screenshots/2_describe_asset_after.png")
        
        # ----------------- STEP 3: Segment Object -----------------
        print("\n=== STEP 3: Segment Object ===")
        # Before Segment: Objects are loaded but no segmentation mask is active
        print("Saving 3_segment_object_before.png...")
        page.screenshot(path="screenshots/3_segment_object_before.png")
        
        # Action: Find the card for 'truck' (or similar object like 'pickup truck' or 'white pickup truck') and click its Segment button
        segment_btn = None
        for object_name in ["truck", "pickup truck", "white pickup truck", "white truck", "vehicle"]:
            h4_element = page.locator(f"h4:has-text('{object_name}')").first
            if h4_element.is_visible():
                print(f"Found object label: '{object_name}'")
                segment_btn = h4_element.locator("xpath=../..").locator("button", has_text="Segment").first
                break
                
        if segment_btn is None:
            # Fallback to the first available Segment button on the page
            print("Fallback: Using the first available Segment button.")
            segment_btn = page.locator("button", has_text="Segment").first
            
        print("Triggering segmentation...")
        segment_btn.click()
        
        # Wait: Wait for SAM segmentation API call to complete
        print("Waiting for SAM3 segmentation to complete and mask to overlay...")
        # When active, the button text toggles to '✓ Active'
        page.wait_for_selector("button:has-text('Active')", timeout=30000)
        page.wait_for_timeout(3000)  # Stabilize mask rendering overlay
        
        # After Segment: The overlay image/canvas shows the highlight mask
        print("Saving 3_segment_object_after.png...")
        page.screenshot(path="screenshots/3_segment_object_after.png")
        
        browser.close()
        print("\nSuccess! E2E screenshots testing completed.")

if __name__ == "__main__":
    main()

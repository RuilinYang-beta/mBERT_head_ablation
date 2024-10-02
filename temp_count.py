from datasets import load_dataset, load_from_disk, DatasetDict

xnli = load_from_disk("./playground/xnli")

total_characters = 0
x = 2500  # Set your desired progress interval

for i, example in enumerate(xnli['train']):
    premise_text = example['premise']['en']
    hypothesis_text = example['hypothesis']['translation'][4]  # Accessing the English translation
    total_characters += len(premise_text) + len(hypothesis_text)
    
    # Print progress every x examples
    if (i + 1) % x == 0:
        print(f"The first {i + 1} examples of training set has total of {total_characters} characters")
        break
        # first 2500 examples has 435220 characters, 
        # translating it to 14 languages (en excluded) will cost
        # (435220 * 14 - 500000) / 1000000 * 20 = 111.8616 dollars 

print("Total number of characters:", total_characters)



# The first 5000 examples of training set has total of 865497 characters
# The first 10000 examples of training set has total of 1732753 characters
# The first 15000 examples of training set has total of 2593856 characters
# The first 20000 examples of training set has total of 3445041 characters
# The first 25000 examples of training set has total of 4297015 characters
# The first 30000 examples of training set has total of 5163301 characters
# The first 35000 examples of training set has total of 6033801 characters
# The first 40000 examples of training set has total of 6886659 characters
# The first 45000 examples of training set has total of 7762081 characters
# The first 50000 examples of training set has total of 8628765 characters
# The first 55000 examples of training set has total of 9483467 characters
# The first 60000 examples of training set has total of 10350869 characters
# The first 65000 examples of training set has total of 11218491 characters
# The first 70000 examples of training set has total of 12089718 characters
# The first 75000 examples of training set has total of 12961240 characters
# The first 80000 examples of training set has total of 13830006 characters
# The first 85000 examples of training set has total of 14691776 characters
# The first 90000 examples of training set has total of 15563474 characters
# The first 95000 examples of training set has total of 16422386 characters
# The first 100000 examples of training set has total of 17288922 characters
# The first 105000 examples of training set has total of 18151408 characters
# The first 110000 examples of training set has total of 19020493 characters
# The first 115000 examples of training set has total of 19894080 characters
# The first 120000 examples of training set has total of 20765555 characters
# The first 125000 examples of training set has total of 21630213 characters
# The first 130000 examples of training set has total of 22496926 characters
# The first 135000 examples of training set has total of 23361347 characters
# The first 140000 examples of training set has total of 24233048 characters
# The first 145000 examples of training set has total of 25099800 characters
# The first 150000 examples of training set has total of 25964000 characters
# The first 155000 examples of training set has total of 26843874 characters
# The first 160000 examples of training set has total of 27711907 characters
# The first 165000 examples of training set has total of 28587733 characters
# The first 170000 examples of training set has total of 29446568 characters
# The first 175000 examples of training set has total of 30309157 characters
# The first 180000 examples of training set has total of 31188812 characters
# The first 185000 examples of training set has total of 32049419 characters
# The first 190000 examples of training set has total of 32911183 characters
# The first 195000 examples of training set has total of 33777685 characters
# The first 200000 examples of training set has total of 34652030 characters
# The first 205000 examples of training set has total of 35523286 characters
# The first 210000 examples of training set has total of 36391394 characters
# The first 215000 examples of training set has total of 37256616 characters
# The first 220000 examples of training set has total of 38128501 characters
# The first 225000 examples of training set has total of 39002651 characters
# The first 230000 examples of training set has total of 39875978 characters
# The first 235000 examples of training set has total of 40734438 characters
# The first 240000 examples of training set has total of 41593765 characters
# The first 245000 examples of training set has total of 42454722 characters
# The first 250000 examples of training set has total of 43327786 characters
# The first 255000 examples of training set has total of 44188561 characters
# The first 260000 examples of training set has total of 45065903 characters
# The first 265000 examples of training set has total of 45940935 characters
# The first 270000 examples of training set has total of 46812488 characters
# The first 275000 examples of training set has total of 47659823 characters
# The first 280000 examples of training set has total of 48526007 characters
# The first 285000 examples of training set has total of 49400919 characters
# The first 290000 examples of training set has total of 50255915 characters
# The first 295000 examples of training set has total of 51118468 characters
# The first 300000 examples of training set has total of 51982712 characters
# The first 305000 examples of training set has total of 52856736 characters
# The first 310000 examples of training set has total of 53737318 characters
# The first 315000 examples of training set has total of 54602732 characters
# The first 320000 examples of training set has total of 55479726 characters
# The first 325000 examples of training set has total of 56355585 characters
# The first 330000 examples of training set has total of 57217272 characters
# The first 335000 examples of training set has total of 58100919 characters
# The first 340000 examples of training set has total of 58974264 characters
# The first 345000 examples of training set has total of 59841553 characters
# The first 350000 examples of training set has total of 60716463 characters
# The first 355000 examples of training set has total of 61587478 characters
# The first 360000 examples of training set has total of 62459154 characters
# The first 365000 examples of training set has total of 63326934 characters
# The first 370000 examples of training set has total of 64196350 characters
# The first 375000 examples of training set has total of 65075394 characters
# The first 380000 examples of training set has total of 65947812 characters
# The first 385000 examples of training set has total of 66809608 characters
# The first 390000 examples of training set has total of 67686667 characters
# Total number of characters: 68155340
# **Text-Based Search Engine and Web Crawler**

This repository combines a web crawler and a text-based search engine. The web crawler fetches posts from specified categories on [Tuổi Trẻ Online](https://tuoitre.vn) and saves them as JSON files. These files are then used by a GUI-based search engine to find posts based on text input.

---

## **Run Code**

1. **Web Crawler**:
   - Run `python crawler.py`.
   - Crawls posts from categories specified in the `crawler.py` file.
   - To Crawl Posts: Modify the category_urls list in (https://tuoitre.vn) (example: https://tuoitre.vn/cong-nghe.html) to specify the categories you want to crawl.
   - Posts are turned into JSON files and stored in the `data/` folder.

2. **GUI Search Engine**:
   - Run `python main.py`.
   - Uses the JSON files in the `data/` folder for a search engine.
   - Allows users to search posts using a GUI.

---

## **Setup**

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

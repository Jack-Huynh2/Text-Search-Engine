import requests
from bs4 import BeautifulSoup
import os
import hashlib
import json
import logging
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)

def fetch_html(url):
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {url}")
    return BeautifulSoup(response.content, "html.parser")

def extract_post_details(post_url):
    try:
        soup = fetch_html(post_url)
        post_id = hashlib.md5(post_url.encode()).hexdigest()
        title = soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else None
        content = '\n'.join(p.get_text() for p in soup.find('div', class_='detail-content').find_all('p')) if soup.find('div', class_='detail-content') else None
        
        if not title or not content:
            logging.warning(f"Skipping post {post_url} as it lacks required fields")
            return None

        author = soup.find('meta', property='dable:author')['content'] if soup.find('meta', property='dable:author') else None
        date = soup.find('meta', property='article:published_time')['content'] if soup.find('meta', property='article:published_time') else None
        category = soup.find('meta', property='article:section')['content'] if soup.find('meta', property='article:section') else None

        post_details = {
            'postId': post_id,
            'title': title,
            'author': author,
            'date': date,
            'category': category,
            'content': content
        }
        return post_details
    except Exception as e:
        logging.error(f"Error extracting post details for {post_url}: {e}")
        return None

def save_post_data(post_data):
    if post_data:
        post_id = post_data['postId']
        directory = 'data2'
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{post_id}.json")
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(post_data, file, ensure_ascii=False, indent=4)

def get_all_posts_from_category(category_url, max_pages=100):
    try:
        post_links = []
        for page in range(1, max_pages + 1):
            paginated_url = f"{category_url.rstrip('.htm')}/trang-{page}.htm" if page > 1 else category_url
            logging.info(f"Fetching page: {paginated_url}")
            response = requests.get(paginated_url, timeout=10)
            if response.status_code != 200:
                logging.error(f"Failed to retrieve {paginated_url}")
                break
            soup = BeautifulSoup(response.content, 'html.parser')
            for link in soup.select(".box-category-item .box-category-link-title"):
                post_url = link.get("href")
                if post_url:
                    full_url = urljoin(category_url, post_url)
                    post_links.append(full_url)
            if not soup.select_one(".pagination-next"):
                break
        return post_links
    except Exception as e:
        logging.error(f"Error getting posts from category {category_url}: {e}")
        return []

def scrape_posts(category_url, max_pages=100):
    scraped_posts = set()
    post_links = get_all_posts_from_category(category_url, max_pages)
    for post_url in post_links:
        try:
            post_data = extract_post_details(post_url)
            if post_data and post_data['postId'] not in scraped_posts:
                scraped_posts.add(post_data['postId'])
                save_post_data(post_data)
                logging.info(f"Saved post: {post_data['postId']}")
            else:
                logging.info(f"Skipping duplicate post: {post_url}")
        except Exception as e:
            logging.error(f"Error scraping {post_url}: {e}")

if __name__ == "__main__":
    category_url = "https://tuoitre.vn/ban-doc.htm"
    scrape_posts(category_url, max_pages=100)

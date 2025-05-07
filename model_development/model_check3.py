import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import whois
import datetime
import re
import socket
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the model using joblib
loaded_model = joblib.load('model development/model.joblib')
# loaded_model = joblib.load('model development/model_original.joblib')

def extract_features(url):
    features = {}
    try:
        # Feature 1: Length of URL
        features['length_url'] = len(url)
        print(f"Length of URL: {features['length_url']}")

        # Feature 2: Length of hostname
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc

        features['length_hostname'] = len(hostname)
        print(f"Parsed hostname: {hostname}")
        print(f"Length of hostname: {features['length_hostname']}")

        # Parse URL
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc
        print(f"Parsed hostname: {hostname}")

        # Feature 3: Presence of IP address
        ip_pattern = re.compile(r'\d+\.\d+\.\d+\.\d+')
        features['ip'] = 1 if ip_pattern.fullmatch(hostname) else 0
        print(f"Presence of IP in hostname: {features['ip']}")

        # Feature 4: Number of dots in URL
        features['nb_dots'] = url.count('.')
        print(f"Number of dots in URL: {features['nb_dots']}")

        # Feature 5: Number of hyphens
        features['nb_hyphens'] = url.count('-')
        print(f"Number of hyphens: {features['nb_hyphens']}")

        # Feature 6: Number of '@' symbols
        features['nb_at'] = url.count('@')
        print(f"Number of '@' symbols: {features['nb_at']}")

        # Feature 7: Number of question marks
        features['nb_qm'] = url.count('?')
        print(f"Number of question marks: {features['nb_qm']}")

        # Feature 8: Number of '&' symbols
        features['nb_and'] = url.count('&')
        print(f"Number of '&' symbols: {features['nb_and']}")

        # Feature 9: Number of slashes
        features['nb_slash'] = url.count('/')
        print(f"Number of slashes: {features['nb_slash']}")

        # Feature 10: Number of semicolons
        features['nb_semicolumn'] = url.count(';')
        print(f"Number of semicolons: {features['nb_semicolumn']}")

        # Feature 11: Presence of "www" in URL
        features['nb_www'] = 1 if "www" in hostname else 0
        print(f"Presence of 'www': {features['nb_www']}")

        # Feature 12: Presence of ".com"
        features['nb_com'] = 1 if ".com" in url else 0
        print(f"Presence of '.com': {features['nb_com']}")

        # Feature 13: HTTPS token
        features['https_token'] = 1 if "https" in hostname else 0
        print(f"Presence of 'https' token: {features['https_token']}")

        # Feature 14: Ratio of digits in URL
        features['ratio_digits_url'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
        print(f"Ratio of digits in URL: {features['ratio_digits_url']}")

        # Feature 15: Ratio of digits in hostname
        features['ratio_digits_host'] = sum(c.isdigit() for c in hostname) / len(hostname) if len(hostname) > 0 else 0
        print(f"Ratio of digits in hostname: {features['ratio_digits_host']}")

        # Feature 16: TLD in subdomain (check for common TLDs in the subdomain part)
        tld_pattern = re.compile(r'\.(com|org|net|info|biz|io)')
        subdomain = hostname.split('.')[0]  # First part of the hostname
        print(f"Subdomain: {subdomain}")
        features['tld_in_subdomain'] = 1 if tld_pattern.search(subdomain) else 0
        print(f"TLD in subdomain: {features['tld_in_subdomain']}")

        # Feature 17: Abnormal subdomain (check if there are more than two subdomains)
        subdomains = hostname.split('.')[:-2]  # Removing the domain and TLD parts
        print(f"Subdomains: {subdomains}")
        features['abnormal_subdomain'] = 1 if len(subdomains) > 2 else 0
        print(f"Abnormal subdomain: {features['abnormal_subdomain']}")

        # Feature 18: Number of subdomains
        subdomains = hostname.split('.')[:-2]  # Remove domain and TLD parts
        features['nb_subdomains'] = len(subdomains)
        print(f"Subdomains: {subdomains}, Number of subdomains: {features['nb_subdomains']}")

        # Feature 19: Prefix-suffix (check for hyphen in hostname)
        features['prefix_suffix'] = 1 if '-' in hostname else 0
        print(f"Prefix-Suffix: {features['prefix_suffix']}")

        # Feature 20: Shortening service (check for common URL shortening services)
        shortening_services = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly"]
        features['shortening_service'] = 1 if any(service in hostname for service in shortening_services) else 0
        print(f"Shortening service detected: {features['shortening_service']}")

        # Feature 21: Length of words in raw URL
        raw_words = re.split(r'[\W_]+', url)
        features['length_words_raw'] = len(raw_words)
        print(f"Words in raw URL: {raw_words}, Length: {features['length_words_raw']}")

        # Feature 22: Shortest word in hostname
        host_words = re.split(r'[\W_]+', hostname)
        features['shortest_word_host'] = len(min(host_words, key=len)) if host_words else 0
        print(f"Words in hostname: {host_words}, Shortest word length: {features['shortest_word_host']}")

        # Feature 23: Longest word in raw URL
        features['longest_words_raw'] = len(max(raw_words, key=len)) if raw_words else 0
        print(f"Longest word in raw URL: {max(raw_words, key=len) if raw_words else ''}, Length: {features['longest_words_raw']}")

        # Feature 24: Longest word in hostname
        features['longest_word_host'] = len(max(host_words, key=len)) if host_words else 0
        print(f"Longest word in hostname: {max(host_words, key=len) if host_words else ''}, Length: {features['longest_word_host']}")

        # Split the raw URL and hostname into words
        raw_words = re.split(r'[\W_]+', url)
        host_words = re.split(r'[\W_]+', hostname)
        path_words = re.split(r'[\W_]+', parsed_url.path)

        # Feature 25: Average word length in raw URL
        features['avg_words_raw'] = sum(len(word) for word in raw_words) / len(raw_words) if raw_words else 0
        print(f"Average word length in raw URL: {features['avg_words_raw']}")

        # Feature 26: Average word length in hostname
        features['avg_word_host'] = sum(len(word) for word in host_words) / len(host_words) if host_words else 0
        print(f"Average word length in hostname: {features['avg_word_host']}")

        # Feature 27: Average word length in path
        features['avg_word_path'] = sum(len(word) for word in path_words) / len(path_words) if path_words else 0
        print(f"Average word length in path: {features['avg_word_path']}")

        # Feature 28: Phishing hints (e.g., "secure", "login", etc.)
        phishing_hints = ["secure", "login", "bank", "account", "verify"]
        features['phish_hints'] = 1 if any(hint in url.lower() for hint in phishing_hints) else 0
        print(f"Phishing hints detected: {features['phish_hints']}")

        # Feature 29: Suspicious TLD
        suspicious_tlds = ["tk", "ga", "cf", "ml", "gq"]
        features['suspecious_tld'] = 1 if any(hostname.endswith(f".{tld}") for tld in suspicious_tlds) else 0
        print(f"Suspicious TLD detected: {features['suspecious_tld']}")

        # Feature 30: Statistical report (hardcoded for now)
        features['statistical_report'] = 0
        print(f"Statistical report placeholder: {features['statistical_report']}")

        # Fetch webpage content
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # Feature 31: Number of hyperlinks
        all_links = soup.find_all('a', href=True)
        features['nb_hyperlinks'] = len(all_links)

        # Feature 32: Ratio of internal hyperlinks
        internal_links = [link for link in all_links if hostname in link['href']]
        features['ratio_intHyperlinks'] = len(internal_links) / len(all_links) if all_links else 0

        # Feature 33: Ratio of external redirection
        external_links = [link for link in all_links if hostname not in link['href']]
        features['ratio_extRedirection'] = len(external_links) / len(all_links) if all_links else 0

        # Feature 34: External favicon
        favicon = soup.find("link", rel="icon") or soup.find("link", rel="shortcut icon")
        favicon_url = favicon['href'] if favicon else ""
        features['external_favicon'] = 1 if favicon and hostname not in favicon_url else 0

        # Feature 35: Links in tags
        tags = ['form', 'iframe', 'input']
        features['links_in_tags'] = len([tag for tag in tags if soup.find(tag)])

        # Feature 36 & 37: Media content
        media_tags = soup.find_all(['img', 'video', 'audio', 'source'])
        internal_media = [media for media in media_tags if hostname in (media.get('src') or "")]
        features['ratio_intMedia'] = len(internal_media) / len(media_tags) if media_tags else 0
        features['ratio_extMedia'] = 1 - features['ratio_intMedia']

        # Feature 38: Safe anchor
        safe_anchors = [anchor for anchor in all_links if not anchor['href'] or anchor['href'] == '#']
        features['safe_anchor'] = len(safe_anchors) / len(all_links) if all_links else 0

        # Feature 39: Empty title
        features['empty_title'] = 1 if not soup.title or not soup.title.string.strip() else 0

        # Feature 40: Domain in title
        features['domain_in_title'] = 1 if hostname in (soup.title.string if soup.title else "") else 0

        # Feature 41: Domain with copyright
        copyright_patterns = ['©', '(c)', 'copyright']
        page_text = soup.get_text().lower()
        features['domain_with_copyright'] = 1 if any(pattern in page_text for pattern in copyright_patterns) else 0


    except Exception as e:
        print(f"Error extracting features: {e}")
        features['length_url'] = -1
        features['length_hostname'] = -1
        features['ip'] = -1
        features['nb_dots'] = -1
        features['nb_hyphens'] = -1
        features['nb_at'] = -1
        features['nb_qm'] = -1
        features['nb_and'] = -1
        features['nb_slash'] = -1
        features['nb_semicolumn'] = -1
        features['nb_www'] = -1
        features['nb_com'] = -1
        features['https_token'] = -1
        features['ratio_digits_url'] = -1
        features['ratio_digits_host'] = -1
        features['tld_in_subdomain'] = -1
        features['abnormal_subdomain'] = -1
        features['nb_subdomains'] = -1
        features['prefix_suffix'] = -1
        features['shortening_service'] = -1
        features['length_words_raw'] = -1
        features['shortest_word_host'] = -1
        features['longest_words_raw'] = -1
        features['longest_word_host'] = -1
        features.update({
            'avg_words_raw': -1,
            'avg_word_host': -1,
            'avg_word_path': -1,
            'phish_hints': -1,
            'suspecious_tld': -1,
            'statistical_report': -1,
        })
        features.update({k: -1 for k in [
            'nb_hyperlinks', 'ratio_intHyperlinks', 'ratio_extRedirection',
            'external_favicon', 'links_in_tags', 'ratio_intMedia',
            'ratio_extMedia', 'safe_anchor', 'empty_title',
            'domain_in_title', 'domain_with_copyright'
        ]})

    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc

        # Feature 42 & 43: Whois Features
        try:
            whois_info = whois.whois(hostname)
            creation_date = whois_info.creation_date
            expiration_date = whois_info.expiration_date

            if isinstance(creation_date, list):  # Handle lists for multiple creation dates
                creation_date = creation_date[0]
            if isinstance(expiration_date, list):  # Handle lists for multiple expiration dates
                expiration_date = expiration_date[0]

            features['domain_registration_length'] = (
                (expiration_date - creation_date).days if creation_date and expiration_date else -1
            )
            features['domain_age'] = (
                (datetime.datetime.now() - creation_date).days if creation_date else -1
            )
        except Exception as e:
            print(f"Whois lookup error: {e}")
            features['domain_registration_length'] = -1
            features['domain_age'] = -1

        # Feature 44: DNS record
        try:
            socket.gethostbyname(hostname)
            features['dns_record'] = 1
        except Exception as e:
            print(f"DNS lookup error: {e}")
            features['dns_record'] = 0

        # Feature 45: Google index
        try:
            google_search = f"https://www.google.com/search?q=site:{url}"
            response = requests.get(google_search, timeout=5)
            features['google_index'] = 1 if "did not match any documents" not in response.text else 0
        except Exception as e:
            print(f"Google index check error: {e}")
            features['google_index'] = -1

        # Feature 46: Page rank (placeholder)
        features['page_rank'] = -1  # Requires external service or API
    except Exception as e:
        print(f"Error extracting Whois features: {e}")
        # Default values for errors
        features.update({
            'domain_registration_length': -1,
            'domain_age': -1,
            'dns_record': 0,
            'google_index': -1,
            'page_rank': -1
        })


    return features


# Function to predict the legitimacy of a URL
def predict_url_legitimacy(url):
    # Extract features
    features = extract_features(url)

    # Convert features to a DataFrame
    feature_names = [
        'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 
        'nb_at', 'nb_qm', 'nb_and', 'nb_slash', 'nb_semicolumn', 
        'nb_www', 'nb_com', 'https_token', 'ratio_digits_url', 'ratio_digits_host', 
        'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix', 
        'shortening_service', 'length_words_raw', 'shortest_word_host', 'longest_words_raw', 
        'longest_word_host', 'avg_words_raw', 'avg_word_host', 'avg_word_path', 
        'phish_hints', 'suspecious_tld', 'statistical_report', 'nb_hyperlinks', 
        'ratio_intHyperlinks', 'ratio_extRedirection', 'external_favicon', 
        'links_in_tags', 'ratio_intMedia', 'ratio_extMedia', 'safe_anchor', 
        'empty_title', 'domain_in_title', 'domain_with_copyright', 'domain_registration_length', 
        'domain_age', 'dns_record', 'google_index', 'page_rank'
    ]

    # Convert to DataFrame, make sure to have the same order of features as the model was trained on
    input_data = pd.DataFrame([features], columns=feature_names)

    print(features)
    # Use the loaded model to make a prediction
    prediction = loaded_model.predict(input_data)

    # Return the prediction result
    # 1: Legitimate, 0: Phishing or fraudulent
    return "Legitimate" if prediction == 1 else "Suspicious"
    # return prediction


# Test with example URLs
url_to_check = "http://appleid.apple.com-app.es/"
# features = extract_features(url_to_check)
result = predict_url_legitimacy(url_to_check)

# Output the features
print("Extracted Features:", result)

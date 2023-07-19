import csv
import re
import time

import requests
import pandas as pd
import json

url = 'https://uk-property-development-finance.com/wp-admin/admin-ajax.php'

headers = {
    'authority': 'uk-property-development-finance.com',
    'accept': '*/*',
    'accept-language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
    'cache-control': 'no-cache',
    'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'cookie': 'cookielawinfo-checkbox-necessary=yes; cookielawinfo-checkbox-functional=yes; cookielawinfo-checkbox-performance=yes; cookielawinfo-checkbox-analytics=yes; cookielawinfo-checkbox-advertisement=yes; cookielawinfo-checkbox-others=yes; CookieLawInfoConsent=eyJuZWNlc3NhcnkiOnRydWUsImZ1bmN0aW9uYWwiOnRydWUsInBlcmZvcm1hbmNlIjp0cnVlLCJhbmFseXRpY3MiOnRydWUsImFkdmVydGlzZW1lbnQiOnRydWUsIm90aGVycyI6dHJ1ZX0=; viewed_cookie_policy=yes; _lscache_vary=246c3110c7c38fb6708f445c2722fb88',
    'origin': 'https://uk-property-development-finance.com',
    'pragma': 'no-cache',
    'referer': 'https://uk-property-development-finance.com/internal-area-search-by-postcode/',
    'sec-ch-ua': '"Google Chrome";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Mobile Safari/537.36',
    'x-requested-with': 'XMLHttpRequest'
}

postcodes = pd.read_csv("ukpostcodes.csv")

extracted_data = []

for postcode in postcodes["postcode"]:
    if re.search("rg6", postcode.lower()):

        data = {
            'postcode': postcode,
            'action': 'internalAreaLookup'
        }

        response = requests.post(url, headers=headers, data=data)

        print(response.text)

        # Extract the response content as a string
        response_content = response.content.decode('utf-8')

        # Parse the JSON response
        json_data = json.loads(response_content)

        # Extract the data field
        data_html = json_data['data']

        # Extract the required information from the HTML
        start_index = data_html.find('<div class="iaf-single-item"')
        end_index = data_html.rfind('</div>')
        data_extracted = data_html[start_index:end_index]

        address_regex = re.compile(r'data-address="([^"]+)"')
        postcode_regex = re.compile(r'data-postcode="([^"]+)"')
        floorarea_regex = re.compile(r'data-floorarea="([^"]+)"')

        # Find all matches for each field
        addresses = re.findall(address_regex, data_extracted)
        postcodes = re.findall(postcode_regex, data_extracted)
        floorareas = re.findall(floorarea_regex, data_extracted)

        # Append the extracted data to the list
        for i in range(len(addresses)):
            extracted_data.append({
                'Address': addresses[i],
                'Postcode': postcodes[i],
                'Floor Area': floorareas[i]
            })

        time.sleep(3)

# Write the extracted data to a CSV file
filename = 'extracted_data.csv'
fieldnames = ['Address', 'Postcode', 'Floor Area']

with open(filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(extracted_data)

print(f"Data written to '{filename}' successfully.")

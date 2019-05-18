# import requests
#
# key = 'AIzaSyAYeCUJwGYBKRdvifnR3ggtuR12t0xe3vA'
# URL = "https://sb-ssl.google.com/safebrowsing/api/lookup?client=api&apikey={key}&appver=1.0&pver=3.0&url={url}"
#
#
# def is_safe(url):
#     response = requests.get("https://sb-ssl.google.com/safebrowsing/api/lookup?client=api&apikey=" + key + "&appver=1.0&pver=3.0&url={url}" + url)
#     print(response.text)
#     return response.text != 'malware'
#
#
# def main():
#     print(is_safe( 'http://addonrock.ru/Debugger.js/'))  # prints False
#     print(is_safe( 'http://google.com'))  # prints True
#
#
# if __name__ == '__main__':
#     main()

import safebrowsing

apikey = 'AIzaSyAYeCUJwGYBKRdvifnR3ggtuR12t0xe3vA'
sb = safebrowsing.LookupAPI(apikey)
resp = sb.threat_matches_find('http://google.com/')
print("matches" in resp and (len(resp["matches"]) > 0))
print (resp)

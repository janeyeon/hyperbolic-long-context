# kill -9 $(nvidia-smi | grep python | awk '{print $5}' | sort -u)


kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u)
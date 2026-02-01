curl -fSL --get \
  --data-urlencode 'query=organism_id:9606' \
  --data 'compressed=true' \
  --data 'format=json' \
  'https://rest.uniprot.org/uniprotkb/stream' \
  -o /home/iscb/wolfson/annab4/DB/all_proteins/human_all.json.gz

todo_list="/home/iscb/wolfson/annab4/catalytic-sites-annotation/isoforms/to_rerun.txt"
pdb_dir="/home/iscb/wolfson/annab4/Data/PDB_Human_Isoforms"
out_dir="/home/iscb/wolfson/annab4/outputs/isoform_groups_todo"

mkdir -p "$out_dir"

while read -r line; do
  base_id="$(echo "$line" | awk '{print $1}')"
  [ -z "$base_id" ] && continue
  [ "${base_id:0:1}" = "#" ] && continue

  out_file="$out_dir/${base_id}_isoforms.txt"
  tmp_file="$out_file.tmp"
  : > "$tmp_file"

  # находим все PDB для base_id
  found=0
  for f in "$pdb_dir/${base_id}-"*.pdb; do
    [ -e "$f" ] || continue
    realpath "$f" >> "$tmp_file"
    found=1
  done

  if [ "$found" -eq 0 ]; then
    echo "WARN: no PDBs for base_id=$base_id" >&2
    rm -f "$tmp_file"
    continue
  fi

  # стабильно сортируем и финализируем
  sort -u "$tmp_file" > "$out_file"
  rm -f "$tmp_file"

  echo "OK: $base_id -> $(wc -l < "$out_file") pdbs"
done < "$todo_list"

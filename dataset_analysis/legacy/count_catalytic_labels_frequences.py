from collections import Counter

def calculate_class_frequencies(file_paths):
    """
    Calculate class frequencies for given file paths.
    :param file_paths: List of file paths.
    :return: Frequencies for each file and overall frequencies.
    """
    overall_counter = Counter()
    results = {}

    for file_path in file_paths:
        counter = Counter()
        with open(file_path, 'r') as file:
            for line in file:
                if not line.startswith('>'):
                    label = line.strip().split()[-1]  # Extract last column
                    counter[label] += 1
        results[file_path] = counter
        overall_counter.update(counter)

    # Calculate frequencies
    frequencies = {file_path: {label: count / sum(counter.values())
                               for label, count in counter.items()}
                   for file_path, counter in results.items()}
    overall_frequencies = {label: count / sum(overall_counter.values())
                           for label, count in overall_counter.items()}

    return frequencies, overall_frequencies


# Paths to your dataset files
file_paths = [
    "/home/iscb/wolfson/annab4/DB/splitted_by_EC_number/train.txt",
    "/home/iscb/wolfson/annab4/DB/splitted_by_EC_number/test.txt",
    "/home/iscb/wolfson/annab4/DB/splitted_by_EC_number/validate.txt"
]

# Calculate frequencies
frequencies, overall_frequencies = calculate_class_frequencies(file_paths)

# Print results
for file_path, freq in frequencies.items():
    print(f"Frequencies in {file_path}:")
    for label, value in freq.items():
        print(f"  Class {label}: {value:.6f}")

print("\nOverall frequencies:")
for label, value in overall_frequencies.items():
    print(f"  Class {label}: {value:.6f}")

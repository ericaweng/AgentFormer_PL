import requests
from bs4 import BeautifulSoup

def load_word_list(file_path):
    with open(file_path, 'r') as f:
        words = f.read().splitlines()
    return words

def get_past_words():
    url = 'https://www.fiveforks.com/wordle/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/94.0.4606.81 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;'
                  'q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,'
                  'application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
        return set()
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
        return set()
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
        return set()
    except requests.exceptions.RequestException as err:
        print("An error occurred:", err)
        return set()

    soup = BeautifulSoup(response.content, 'html.parser')

    past_words = set()
    for word in [a.split(" ")[0] for a in str(soup.find('div', id='vlist')).splitlines()]:
        if len(word) == 5 and word.isalpha():
            past_words.add(word.lower())
    return past_words

def filter_words(words, correct_positions, incorrect_positions, incorrect_letters, required_letters):
    filtered_words = []
    for word in words:
        match = True

        # Check for correct letters in correct positions (green letters)
        for idx, letter in correct_positions.items():
            if word[idx] != letter:
                match = False
                break
        if not match:
            continue

        # Check for incorrect letters in certain positions (yellow letters)
        for idx, letters in incorrect_positions.items():
            if word[idx] in letters:
                match = False
                break
        if not match:
            continue

        # Ensure required letters are in the word (from yellow letters)
        if not all(letter in word for letter in required_letters):
            continue

        # Exclude words containing incorrect letters (grey letters)
        if any(letter in word for letter in incorrect_letters):
            continue

        filtered_words.append(word)
    return filtered_words


def load_frequency_list(freq_file_path):
    frequency_dict = {}
    with open(freq_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0].lower()
                freq = float(parts[1])
                frequency_dict[word] = freq
    return frequency_dict



def get_feedback(guess, solution):
    """
    Returns a tuple representing the feedback pattern:
    2 for green (correct position),
    1 for yellow (incorrect position),
    0 for grey (not in word).
    """
    feedback = [0] * 5
    guess_remaining = []
    solution_remaining = []

    # First pass: identify greens
    for i in range(5):
        if guess[i] == solution[i]:
            feedback[i] = 2
        else:
            guess_remaining.append((i, guess[i]))
            solution_remaining.append(solution[i])

    # Second pass: identify yellows
    for idx, letter in guess_remaining:
        if letter in solution_remaining:
            feedback[idx] = 1
            solution_remaining.remove(letter)
    return tuple(feedback)

def calculate_word_effectiveness(possible_words):
    """
    Calculates the expected number of remaining words after guessing each word.
    Returns a dictionary mapping words to their effectiveness scores.
    Lower scores are better (more effective).
    """
    effectiveness_scores = {}
    total_possible_words = len(possible_words)
    for guess_word in possible_words:
        pattern_counts = {}
        for solution_word in possible_words:
            feedback_pattern = get_feedback(guess_word, solution_word)
            pattern_counts[feedback_pattern] = pattern_counts.get(feedback_pattern, 0) + 1
        # Expected remaining words is the sum of (count^2 / total_possible_words)
        expected_remaining = sum(count ** 2 for count in pattern_counts.values()) / total_possible_words
        effectiveness_scores[guess_word] = expected_remaining
    return effectiveness_scores


def main():
    # Load word list
    words = load_word_list('nyt-answers-alphabetical.txt')

    # Load frequency list
    frequency_dict = load_frequency_list('en_full.txt')  # Update with your frequency file path
    
    # Get past words and remove them
    past_words = get_past_words()

    # Initialize dictionaries and lists for user inputs
    correct_positions = {}       # e.g., {0: 'c', 3: 'a'}
    incorrect_positions = {}     # e.g., {1: ['a', 'e'], 2: ['o']}
    incorrect_letters = []       # e.g., ['b', 'd', 'f']
    required_letters = []        # Letters that must be in the word (from yellow letters)

    # Collect correct letters (green)
    print("Enter correct letters (green) with positions (e.g., 'c1 a4' for 'c' at position 1, 'a' at position 4):")
    correct_input = input()
    for item in correct_input.strip().split():
        if len(item) >= 2:
            letter = item[0].lower()
            position = int(item[1])  # Convert to zero-based index
            correct_positions[position] = letter

    # Collect incorrect letters in certain positions (yellow)
    print("Enter incorrect letters in certain positions (yellow), format 'a1 e2' for letters and positions:")
    incorrect_input = input()
    for item in incorrect_input.strip().split():
        if len(item) >= 2:
            letter = item[0].lower()
            position = int(item[1])  # Convert to zero-based index
            incorrect_positions.setdefault(position, []).append(letter)
            if letter not in required_letters:
                required_letters.append(letter)

    # Collect incorrect letters (grey)
    print("Enter incorrect letters (grey):")
    incorrect_letters = input()

    # Filter words based on input
    possible_words = filter_words(words, correct_positions, incorrect_positions, incorrect_letters, required_letters)

    # remove words that are not in the past words
    possible_words = [word for word in possible_words if word not in past_words]

    # Sort possible words by frequency
    possible_words_sorted = sorted(possible_words, key=lambda w: frequency_dict.get(w, 0), reverse=True)

    # Output possible words
    print("\nPossible words sorted by frequency:")
    for word in possible_words_sorted:
        freq = frequency_dict.get(word, 'N/A')
        print(f"{word} - Frequency: {freq}")
    # Calculate effectiveness scores
    effectiveness_scores = calculate_word_effectiveness(possible_words)

    # Combine frequency and effectiveness into one list
    words_with_scores = [
        (word, frequency_dict.get(word, 0), effectiveness_scores[word])
        for word in possible_words_sorted
    ]

    # Output possible words with frequency and effectiveness
    print("\nPossible words sorted by effectiveness (lower is better):")
    for word, freq, effectiveness in sorted(words_with_scores, key=lambda x: (effectiveness_scores[x[0]], -frequency_dict.get(x[0], 0))):
        print(f"{word} - Frequency: {freq}, Expected Remaining Words: {effectiveness:.2f}")
        

if __name__ == "__main__":
    main()

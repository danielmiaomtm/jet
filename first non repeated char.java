public char firstNotRepeatedChar (String s) {
	char[] strChars = s.toCharArray();
	HashMap<Character, Integer> charMap = new HashMap<Character, Integer>();
	Queue<Character> strQueue = new LinkedList<Character>();
	for(int i = 0; i < strChars.length; i++) {
		if(charMap.containsKey(strChars[i])) {
			charMap.put(strChars[i], charMap.get(strChars[i]) + 1);
		}
		else {
			charMap.put(strChars[i], 1);
			strQueue.add(strChars[i]);
		}
	}

	while(!strQueue.isEmpty()) {
		char firstUnique = strQueue.poll();
		if(charMap.get(firstUnique) == 1) {
			return firstUnique;
		}
	}
	return null;
}

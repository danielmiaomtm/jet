	public String compressedString (String input) {
		int count = 1;
		char pre = input.charAt(0);
		StringBuilder sb = new StringBuilder();
		for (int i = 1; i < input.length(); i++) {
			if (input.charAt(i) == pre) {
				count++;
				continue;
			} else {
				sb.append(pre).append(count);
				pre = input.charAt(i);
				count = 1;
			}
		}
		sb.append(pre).append(count);
		return sb.toString();
	}
	
	
	public String compressedString2 (String input) {
		if (input == null || input.length() < 2) {
			return input;
		}
		int len = input.length();
		String result = helper(input, 1, input.charAt(0), 1, input.charAt(0) + "");
		if (result.length() < len) {
			return result;
		}
		return input;
	}
	public String helper (String input, int index, char pre, int count, String output) {
		if (index == input.length()) {
				output += count;
				return output;
		}
		if (input.charAt(index) == pre) {
			return helper(input, index + 1, pre, count + 1, output);
		} else {
			return helper(input, index + 1, input.charAt(index), 1, output + count + input.charAt(index));
		}
	}

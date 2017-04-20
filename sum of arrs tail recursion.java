public int sumOfArra (int[] arrs) {
		if (arrs == null || arrs.length == 0) {
			return 0;
		}
		return helper(arrs, 0, 0);
	}
	public int helper (int[] arrs, int index, int sum) {
		if (index == arrs.length) {
			return sum;
		}
		return helper(arrs, index + 1, sum + arrs[index]);
	}

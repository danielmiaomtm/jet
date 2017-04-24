map : for each

nums.(num) -> num * num;

public String[] transToCap (int[] nums) {
	String[] result = new String[nums.length];
	helper(result, nums, 0);
	return result;
}
public void helper (String[] result, int[] nums, int index) {
	if (index >= nums.length) {
		return;
	}
	result[index] = String.valueOf(nums[index]);
	helper(result, nums, index + 1);
}	 


reduce : sum of arrs 

public int SumUp (int[] nums) {
	return helper(nums, 0, 0);
}
public int helper (int[] nums, int index, int result) {
	if (index == nums.length) {
		return result;
	}
	return helper(nums, index + 1, result + nums[index]);
}

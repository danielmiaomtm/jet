/*
Given a sorted integer array where the range of elements are in the inclusive range [lower, upper], return its missing ranges.

For example, given [0, 1, 3, 50, 75], lower = 0 and upper = 99, return ["2", "4->49", "51->74", "76->99"].
*/


public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> result = new ArrayList<>();
        
        int pre = lower - 1;
        for (int i = 0; i <= nums.length; i++) {
            int cur = i == nums.length ? upper + 1 : nums[i];
            if (pre + 2 == cur) {
                result.add(String.valueOf(pre + 1));
            } else if (pre + 2 < cur) {
                result.add(String.valueOf(pre + 1) + "->" + String.valueOf(cur - 1));
            }
            pre = cur;
        }
        
        return result;
    }

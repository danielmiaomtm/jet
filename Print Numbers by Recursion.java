/*
Print numbers from 1 to the largest number with N digits by recursion.

 Notice

It's pretty easy to do recursion like:

recursion(i) {
    if i > largest number:
        return
    results.add(i)
    recursion(i + 1)
}
however this cost a lot of recursion memory as the recursion depth maybe very large. Can you do it in another way to recursive 
with at most N depth?

Have you met this question in a real interview? Yes
Example
Given N = 1, return [1,2,3,4,5,6,7,8,9].

Given N = 2, return [1,2,3,4,5,6,7,8,9,10,11,12,...,99].
*/ 


public class Solution {
    /**
     * @param n: An integer.
     * return : An array storing 1 to the largest number with n digits.
     */
    public List<Integer> numbersByRecursion(int n) {
        // write your code here
        List<Integer> res = new ArrayList<Integer>();
        if (n >= 0) {
            helper(n, res);
        }
        return res;
    }
    public int helper(int n, List<Integer> res) {
        if (n == 0) {
            return 1;
        }
        int base = helper(n - 1, res);
        int size = res.size();
        for (int i = 1; i <= 9; i++) {
            int curbase = i * base;
            res.add(curbase);
            for (int j = 0; j < size; j++) {
                res.add(curbase + res.get(j));
            }
        }
        return base * 10;
    }
}

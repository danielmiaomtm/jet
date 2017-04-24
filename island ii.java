public class Solution {
    int totalIsland = 0;
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> result = new ArrayList<>();
        int[] arrs = new int[m * n];
        Arrays.fill(arrs, -1);
        
        for (int i = 0; i < positions.length; i++) {
            helper(m, n, positions[i], arrs);
            result.add(totalIsland);
        }
        return result;
    }
    public void helper (int m, int n, int[] position, int[] arrs) {
        int[][] dir = new int[][]{{0,1},{1,0},{-1,0},{0,-1}};
        int i = position[0], j = position[1];
        arrs[i * n + j] = i * n + j;
        totalIsland++;
        
        for (int k = 0; k < dir.length; k++) {
            int row = i + dir[k][0];
            int col = j + dir[k][1];
            if (row >= 0 && row < m && col >= 0 && col < n && arrs[row * n + col] != -1) {
                union(arrs, row * n + col, i * n + j);    
            }
        }
        
    }
    
    public void union (int[] arrs, int i, int j) {
        int num1 = find (arrs, i);
        int num2 = find(arrs, j);
        if (num1 != num2) {
            arrs[num1] = num2;
            totalIsland--;
        }
    }
    
    public int find (int[] arrs, int i) {
        if (i == arrs[i]) {
            return arrs[i];
        }
        arrs[i] = find(arrs, arrs[i]);
        return arrs[i];
    }
    
    
}

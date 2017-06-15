becafb





public static List<List<Integer>> splitList (List<Integer> input) {
	List<List<Integer>> result = new ArrayList<>();
	Map<Integer, Integer> map = new HashMap<>();
	int maxLen = input.size();
	int maxVal = -1;
	for (int i = 0; i < input.size(); i++) {
		int curNum = input.get(i);
		if (!map.containsKey(curNum)) {
			map.put(curNum, 1);
		} else {
			map.put(curNum, map.get(curNum) + 1);;
		}
	}
	PriorityQueue<Map.Entry<Integer, Integer>> heap = new PriorityQueue<>(11, new Comparator<Map.Entry<Integer, Integer>>() {
		@Override
		public int compare (Map.Entry<Integer, Integer> entry1, Map.Entry<Integer, Integer> entry2) {
			return entry2.getValue() - entry1.getValue();
		}
	});
	for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
		heap.offer(entry);
	}
	while (!heap.isEmpty()) {
		Map.Entry<Integer, Integer> cur = heap.poll();
			for (int i = 0; i < cur.getValue(); i++) {
				List<Integer> list;
				if (result.size() < i) {
					list = new ArrayList<>();
				} else {
					list = result.get(i);
				}	
				list.add(cur.getKey());
			}		
	}
	return result;
}











class Node<K,V> {
	Node pre;
	Node next;
	V val;
	K key;
	long time;
	Node (K key, V val, long time) {
		this.key = key;
		this.val = val;
		this.pre = null;
		this.next = null;
		this.time = time;
	}
}


public class ExpiredMap<K, V> {
	Map<K, Node> map = new HashMap<>();
	Node head =  new Node(0, null, null, 0)
	Node tail = new Node(0, null, null, 0);

	public void put (K key, V val, long duration) {
		long curTime = System.currentTimeMillis();

		checkExpired(head, curTime);

		Node temp = new Node(k, val, curTime + duration);
		if (!map.containsKey(key)) {
			//x-y-tail
			moveToTail(temp, tail);
		} else {
			node.pre.next = node.next;
			node.next.pre = node.pre;
			moveToTail(temp, tail);
		}

		map.put(key, temp);

	}


	public void moveToTail (Node node, Node tail) {
		tail.pre.next = temp;
		temp.next = tail;
		temp.pre = tail.pre;
		tail.pre = temp;
	}


	public V get (K key) {		
		long curTime = System.currentTimeMillis();

		checkExpired(head, curTime);
		return map.containsKey(key) ? map.get(key).val : null;
	}

	public void cleanExpired (Node head, long curTime) {
		Node node = head.next;
		while (node != tail && node.time < curTime) {
			map.remove(node.key);
			head.next = node.next;
			node.next.pre = head;
			node.pre = null;
			node.next = null;
			node = head.next;
		}
	}
}











class Node {
	long time;
	K key;
	Node (K key, long time) {
		key = key;
		time = time;
	}
}
public class ExpiredMap<K, V> {
	Map<K, V> map = new HashMap<>();
	Map<K, long> timeMap = new HashMap<>();
	PriorityQueue<Map.Entry<K, long>> heap = new PriorityQueue<>(11, new Comparator<Map.Entry<K, long>> () {
		public int compare (Map.Entry<K, long> entry1, Map.Entry<K, long> entry2) {
			return entry2.getValue() - entry1.getValue();
		}
	});

	public void put (K key, V val, long duration) {
		long curTime = System.currentTimeMillis();
		//clean up the expired keys
		while (!heap.isEmpty() && heap.peek().getValue() < curTime) {
			Node temp = heap.poll();
			map.remove(temp.key);
		}
		map.put(key, val);
		timeMap.put(key, curTime + duration);
	}

	public V get (K key) {
		long curTime = System.currentTimeMillis();
		while (!heap.isEmpty() && heap.peek().time < curTime) {
			Node temp = heap.poll();
			map.remove(temp.key);
		}
		if (!map.containsKey(key)) {
			return null;
		}
		return map.get(key);
	}

	//ex :
	//put(k, v, 1000)
	//get(k) -> v (less than 1000ms has passed since put)
	//get(k) -> null (more than 1000ms has passed since put)
}


public List<Integer> findConsensus(List<Iterator> streams, int m) {
	List<Integer> result = new ArrayList<>();
	if (streams == null || streams.length == 0) {
		return result;
	}

	PriorityQueue<Iterator> heap = new PriorityQueue<Iterator>(11, new Comparator<Iterator>() {
		public int compare (Iterator i1, Iterator i2) {
			return i1.peek() - i2.peek();
		}
	});

	for (Iterator stream : streams) {
		if (stream.hasNext()) {
			heap.offer(stream);
		}
	}

	while (!heap.isEmpty()) {

		Iterator cur = heap.poll();
		int curVal = cur.peek();
		int count = 1;

		// skip the duplicates, and put the udpate iterator into heap
		while (cur.hasNext()) {
			if (curVal == cur.peek()) {
				cur.next();
			} else {
				break;
			}
		}

		if (cur.hasNext()) {
			heap.offer(cur);
		}


		// find next same val
		while (!heap.isEmpty() && curVal == heap.peek().peek()) {
			count++;
			Iterator temp = heap.poll();
			while (temp.hasNext()) {
				if (temp.peek() == curVal) {
					temp.next();
				}  else {
					break;
				}
			}
			if (temp.hasNext()) {
				heap.offer(temp);
			}
		}

		if (count >= m) {
			result.add(curVal);
		}
	}
	return result;
}






public List<Integer> getNumberInAtLeastKStream(List<Stream> lists, int k){
        List<Integer> res = new ArrayList<>();
        if (lists == null || lists.size() == 0) return res;

        PriorityQueue<Num> minHeap = new PriorityQueue<>(new Comparator<Num>() {
            @Override
            public int compare(Num o1, Num o2) {
                return o1.val - o2.val;
            }
        });
        //先把所有的stream放进heap里面
        for (Stream s: lists) {
            if (s.move()){ //这里先判断一下要不就崩了
                minHeap.offer(new Num(s));
            }
        }

        while (!minHeap.isEmpty()){
            Num cur = minHeap.poll();
            int curValue = cur.val;
            int count = 1;
            
            while (cur.stream.move()){
                int nextVal = cur.stream.getValue();
                if (nextVal == curValue){
                    continue;
                }
                else {
                    cur.val = nextVal;
                    minHeap.offer(cur);
                    break;
                }
            }


            //更新其他stream的头部，就是把指针往后挪，相同的数字就计数了。
            while (!minHeap.isEmpty() && curValue == minHeap.peek().val){
                count++;
                Num num = minHeap.poll();
//                int numVal = num.val;

                while (num.stream.move()){
                    int nextVal = num.stream.getValue();
                    if (curValue == nextVal){
                        continue;
                    }
                    else {
                        num.val = nextVal;
                        minHeap.offer(num);
                        break;
                    }
                }
            }



            if (count >= k){
                res.add(curValue);
            }
        }


        return res;
    }

















boolean isSame(TreeNode r1, TreeNode r2) {
	if (r1 == null && r2 == null) {
		return true;
	} else if (r1 == null || r2 == null) {
		return false;
	} else if (r1.val != r2.val) {
		return false;
	}
	return isSame (r1.left, r2.left) && isSame (r2.right, r2.right);
}

public boolean isSubtree (TreeNode r1, TreeNode r2) {
	if (isSame(r1, r2)) {
		return true;
	}
	isSubtree(r1.left, r2) || isSubtree(r2.right, r2);

}


public List<String> strPermuation (String input) {
	List<String> result = new ArrayList<>();
	if (input == null || input.length() == 0) {
		return result;
	}
	Arrays.sort(input);
	char[] chars = input.toCharArray();
	boolean isFinished = false;

	while (!isFinished) {
		result.add(new String(chars));
		int i;
		for (i = chars.size() - 2; i >= 0; i--) {
			if (chars[i] < chars[i + 1]) {
				break;
			}
		}
		int nextLarger = findNextLarger(chars, chars[i], i + 1, chars.size() - 1);
		swap(chars, i, findNext);
		Arrays.sort(chars, i + 1, chars.size() - 1);
	}
	return result;
}
public int findNextLarger (char[] chars, char pre, int left, int right) {
	int result = left;
	for (int i = left + 1; i <= right; i++) {
		if (chars[i] > pre && chars[i] < chars[result]) {
			result = i;
		}
	}
	return result;
}
public void swap (char[] chars, int left, int right) {
	char temp = chars[left];
	chars[left] = chars[right];
	chars[right] = temp;
}


public List<List<Integer>> pa (int input) {
	helper(result, list, input,  0, input);
}
public void helper (result, list, input, curSum, index) {
	if (curSum == 0) {

	}

	for (int i = index; i >= 1; i--) {	
		if (i == input) {
			result.add(new ArrayList<>(input));
		} else {
			list.add(i);
			
			helper(result, list, input, curSum + i, i);
			list.remove)()
	}	
}


public double mysqt (Double input, int x) {
	if (input < 0) {
		return 
	}
	double diff = 1 / Math.pow(10, x);
	int start = 0;
	int end = input < 1 ? 1.0 : input;

	while (start + diff < end) {
		double mid 
		if (mid = input / mid) {
			return mid;
		} else if (mid > input / mid) {
			end = mid;
		} else {
			start = mid;
		}
	}
	return (start + end) / 2;
}




public double myPow (double x, int n) {
	if (n == 0) {
		return 1.0d;
	}
	if (n < 0) {
		if (n == Integer.MIN_VALUE) {
			n++;
			return 1 / (myPow(x, Integer.MAX_VALUE) * x);
		}
		n = -n;
		x = 1 / x;
	}

	double temp = myPow(x, n / 2);
	if (n % 2 == 0) {
		return temp * temp;
	} else {
		return temp * temp * x;
	}

}





public List<Integer> boundaryTraverse (TreeNode root) {
	List<Integer> result = new ArrayList<>();

	List<Integer> temp = new ArrayList<>();
	helper(root, result, true);
	

	List<Integer> rightList = new Arraylist<>();
	getRight (root.right, rightList);
	Collections.reverse(rightList);

	result.addAll(rightList);
	result.addAll(temp);

	return result;
}
public void getLeft (TreeNode root, List<Integer> list) {
	if (root == null) {
		return;
	}
	if (root.left == null && root.right == null) {
		return;
	} else {
		list.add(root.val);
	}
	
	if (root.left != null) {
		getLeft(root.left, list);
	} else if (root.right != null) {
		getLeft(root.right, list);
	}

}
public void getRight (TreeNode root, List<Integer> list) {
	if (root == null) {
		return;
	}
	if (root.left == null && root.right == null) {
		return;
	} else {
		list.add(root.val);
	}
	if (root.right != null) {
		getRight(root.right, list);
	} else if (root.left != null) {
		getRight(root.left, list);
	}
}
public void getMiddle (TreeNode root, List<Integer> list) {
	if (root == null) {
		return;
	}
	if (root.left == null && root.right == null) {
		list.add(root.val);
		return;
	}
	getMiddle(root.left, list);
	getMiddle(root.right, list);
}



//有parent
public TreeNode nextLargestBST (TreeNode node) {
	if (node == null) {
		return null;
	}
	if (node.right != null) {
		return getNext(node.right);
	}
	TreeNode parent = node.parent;
	while (parent != null && parent.val < node.val) {
		parent = parent.parent;
	}
	return parent;
}

//没有parent

public TreeNode nextLargestBSTII (TreeNode root, TreeNode node) {
	if (node == null) {
		return null;
	}
	if (node.right != null) {
		return getNext(node.right);
	}
	TreeNode temp = null;
	while (root != null) {
		if (root.val == node.val) {
			break;
		} else if (root.val < node.val) {
			root = root.right;
		} else {
			root = root.left;
			temp = root;
		}
	}
	return temp;
}




public TreeNode getNext (TreeNode node) {
	TreeNode root = node;
	while (root.left != null) {
		root = root.left;
	}
	return root;
}




public static Integer calculator2 (Stirng input) {

}

TreeNod
public TreeNode cloneGraph (TreeNode root) {

}



public static Integer calculator (String input) {
	input = input.replaceAll(" ", "");
	if (input == null || input.length() == 0) {
		return null;
	}
	int num = 0;
	char pre_sign = '+';

	Stack<Integer> stack = new Stack<>();

	for (int i = 0; i < input.length(); i++) {
		char c = input.charAt(i);
		if (Character.isDigit(c)) {
			num = num * 10 + (c - '0');
		}
		if (!Character.isDigit(c) || i == input.length() - 1) {
			if (pre_sign == '+') {
				stack.push(num);
			} else if (pre_sign == '-') {
				stack.push(-num);
			} else if (pre_sign == '*') {
				int temp = stack.pop();
				stack.push(temp * num);
			} else if (pre_sign == '/') {
				int temp = stack.pop();
				stack.push(temp / num);
			}
			num = 0;
			pre_sign = c;
		}
	}
	int result = 0;
	while (!stack.isEmpty()) {
		result += stack.pop();
	}
	return result;
}




public boolean canIwin () {

}

public String combStr (String input, int target) {

}


public int maxVal (TreeNode root) {
	if (root == null) {
		return 0;
	}
	return helper(root, 0);
}
public void helper (TreeNode root, int curSum) {
	if (root == null) {
		return curSum;
	}
	int left = helper(root.left, curSum + root.val);
	int right = helper (root.right, curSum + root.val);
	int max = root.val;
	return Math.max(Math.max(Math.max(max + left), Math.max(max + right)), max + left + right)
}



//O(n) bucket 
public static List<Integer> kth(int[] arr, int k){
	Map<Integer, Integer> occurence = new HashMap<>();
	int maxCount = 0;
	for (int i = 0; i < arr.length; i++) {
		map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);
		if (maxCount < map.get(arr[i])) {
			maxCount = map.get(arr[i]);
		}
	}
	List<Integer>[] list = new List[maxCount + 1];
	
	for (int key : occurence.keySet()) {
		if (list[occurence.get(key)] == null) {
			list[occurence.get(key)] = new ArrayList<>();
		}
		list[occurence.get(key)].add(key);
	}

	List<Integer> result = new ArrayList<>();
	for (int i = maxCount - 1; i >= 0 && result.size() < k; i--) {
		if (list[i] != null) {
			result.addAll(list[i]);
		}
	}
	return result;
}



//nlgk minHeap

public static List<Integer> kth(int[] arr, int k){
	Map<Integer, Integer> occurence = new HashMap<>();
	for (int i = 0; i < arr.length; i++) {
		map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);
	}
	PriorityQueue<Map.Entry<Integer, Integer>> minHeap = new PriorityQueue<>(11, new Comparator<Map.Entry<Integer, Integer>>() {
		public int compare (Map.Entry<Integer, Integer> entry1, Map.Entry<Integer, Integer> entry2) {
			return entry1.getValue() - entry2.getValue();
		}
	});

	for (Map.Entry<Integer, Integer> entry : occurence) {
		minHeap.offer(entry);
	}

	List<Integer> res = new ArrayList<>();
	while (res.size() < k) {
		Map.Entry<Integer, Integer> entry = minHeap.p
	}
	return res;
}



public int[] slidWin (int[] nums, int k) {
	int[] result = new int[nums.length - k + 1];
	Deque<Integer> deque = new LinkedList<>();
	
	int index = 0;
	for (int i = 0; i < nums.length; i++) {
		while (!deque.isEmpty && nums[deque.peekLast()] >= nums[i]) {
			deque.pollLast();
		}
		while (!deque.isEmpty() && deque.peekFirst() <= i - k) {
			deque.pollFirst();
		}
		deque.offer(nums[i]);
		if (i + 1 - k >= 0) {
			result[i + 1 - k] = nums[deque.peekFirst()];
		}
	}
	return result;
}



public boolean wordBreak (String word, Set<String> dict) {
	boolean[] visited = new boolean[word.length() + 1];
	for (int i = 1; i <= word.length(); i++) {
		for (int j = 0; j < i; j++) {
			if (visited[j] && dict.contains(word.substirng(j, i))) {
				visited[i] = true;
				break;
			}
		}
	}
	return visited[word.length()];
}





public int firstOccurBS (int[] nums, int target) {

}

public void sortColor (int[] nums) {
	if (nums == null || nums.length == 0) {
		return;
	}
	int left = 0, right = nums.length - 1;
	int index = 0;
	while (index < nums.length) {
		if (nums[index] == 1) {
			index++;
		} else if (nums[index] == 0) {
			swap(nums, index++, left++);
		} else if (nums[index] == 2) {
			swap(nums, index, right);
		}
	}
}
public void swap (int[] nums, int p1, int p2) {
	int temp = nums[p1];
	nums[p1] = nums[p2];
	nums[p2] = temp;
}

public TreeNode lca (TreeNode root, TreeNode r1, TreeNode r2) {
	if (root == null) {
		return null;
	}
	if (root == r1 || root == r2) {
		return root;
	}
	TreeNode left = lca(root.left, r1, r2);
	TreeNode right = lca(root.right, r1, r2);
	if (left != null && right != null) {
		return root;
	} else if (left != null) {
		return left;
	} else if (right != null) {
		return right;
	}
}



//a?r?1:b:c
public TreeNode strToTree(String str) {
	if (str == null || str.length() == 0) {
		return null;
	}
	if (str.length() == 1) {
		return new TreeNode(str);
	}
	int flag = 0, mid = 0;
	for (int i = 2; i < str.length(); i++) {
		char c = str.charAt(i);
		if (c == '?') {
			flag++;
		} else if (c == ':') {
			if (flag == 0) {
				mid = i;
				break;
			} else {
				flag--;
			}
		}
	}

	TreeNode root = new TreeNode(s.charAt(0));
	root.left = strToTree(str.substring(2, mid);
	root.right = strToTree(mid + 1);
	return root;
}

public String treeToStr (TreeNode root) {
	if (root == null) {
		return "";
	}
	if (root.left == null && root.right == null) {
		return root.val;
	}
	String left = treeToStr(root.left);
	String right = treeToStr(root.right);

	return root.val + "?" + left + ":" + right;

}









public int strStr (String l, String s) {
	if (l == null || l.length() < s.length()) {
		return -1;
	}
	if (s == null || s.length() == 0) {
		return 0;
	}
	for (int i = 0; i <= l.length() - s.length(); i++) {
		if (l.substring(i, i + s.length()).equals(s)) {
			return i;
		}
	}
	return -1;
}



int sum = 0;
List<Node> result = new ArrayList<>();
public List<Node> getPath (Node root) {
	if (root == null) {
		return result;
	}
	List<Node> list = new ArrayList<>();
	helper (root, 0, list);
	return result;
}
public void helper (Node root, int curSum, List<Node> list) {
	if (root == null) {
		return;
	}
	if (root.left == null && root.right == null) {
		if (curSum + root.val > sum) {
			sum = curSum + root.val;
			result = new ArrayList<>(list);
			result.add(root);
			return;
		}
		return;
	}
	list.add(root);
	helper(root.left, curSum + root.val, list);
	helper(root.right, curSum + root.val, list);
	list.remove(list.size() - 1);
}




int minVal = Integer.MAX_VALUE;
public int minVal (TreeNode root) {
	if (root == null) {
		return -1;
	}
	helper (root.left, root.val);
	helper(root.right, root.val);
	return minVal == Integer.MAX_VALUE ? -1: minVal;
}
public void helper (TreeNode node, int curSum) {

}


public List<String> allPalindromeStr (String input) {
	List<String> result = new ArrayList<>();
	int len = input.length();
	if (len == 0) {
		return result;
	}
	Map<Character, List<Integer>>
	for (int i = 1; i < len; i++) {

	}
}
public void helper () {

}
public boolean isPalindrome () {

}


public void List<List<Integer>> groupId (String[][] pairs, int n) {
	
}
class UF {
	String[] nums;
	UF (int n) {
		nums = new String[n];
		for (int i = 0; i < n; i++) {
			nums[i] = i;
		}
	}
	public void union (int n1, int n2) {
		int f1 = find(n1);
		int f2 = find(n2);
		if (f1 != f2) {
			nums[f1] = f2;
		}
	}
	public int find (int num) {
		if (num == nums[num]) {
			return num;
		}
		int f = find(nums[num]);
		nums[num] = f;
		return f;
	}
}

public List<String> addOperators(String num, int target) {
	List<String> result = new ArrayList<>();
	helper(result, target, num, 0, 0, 0, "");
	return result;
}
public void helper (List<String> result, int target, String num, int index, long pre, long cur, String str) {
	if (index == num.length() && cur == target) {
		result.adD(new String(str));
		return;
	}
	for (int i = 1; i <= num.length(); i++) {
		String temp = num.substring(index, i);
		if (temp.length() > 1 && temp.charAt(0) == '0') {
			return;
		}
		long curVal = Long.parseLong(temp);
		helper(result, target, num, i, curVal, pre + preVal, str + "+" + temp);
		helper(result, target, num, i, -curVal, pre - preVal, str + "-" + temp);
		helper(result, target, num, i, pre * curVal, (cur - preVal) + curVal * pre, str + "*" + temp);
	}
	
}




public List<String> wordLadder (List<String> dict, String start, String end) {
	List<String> result = new ArrayList<>();
	helper(result, dict, start, end);
	return result;
}
public boolean helper () {
	if (start.equals(end)) {
		result.add(end);
		return true;
	}
	char[] chars = start.toCharArray();
	for (int i = 0; i < start.length(); i++) {
		for (char c = 'a'; c <= 'z'; c++) {
			chars[i] = c;
			String temp = new String(chars);
			result.add(start);
			boolean res = helper(result, dict, temp, end);
			if (res) {
				return true;
			}
			result.remove(result.size() - 1);
		}
	}
	return false;
}


public void flattenList(Node node) {
	if (node != null) {
		return;
	}
	Node head = node, tail = null;
	while (node.next != null) {
		node = node.next;
	}
	tail = node;
	Node result = head;
	while (head != tail) {
		if (head.down != null) {
			tail.next = head.down;
			head.down = null;
			while (tail.next != null) {
				tail = tail.next;
			}
		}
		head = head.next;
	}
	
}




public boolean isIsomorphic(String s, String t) {
	if (s.length() != t.length()) {
		return false;
	}
	Map<Character, Character> map = new HashMap<>();
	Set<Character> set = new HashSet<>();
	for (int i = 0; i < s.length(); i++) {
		char sc = s.charAt(i), tc = t.charAt(i);
		if (!map.containsKey(sc)) {
			if (set.contains(tc)) {
				return false;
			}	
			map.put(sc, tc);
		} else {
			if (map.get(sc) != tc) {
				return false;
			}
		}
		set.add(tc);
	}
	return true;
}



public List<String> groupStr (List<String> words) {
	Map<String, List<String>> map = new HashMap<>();
	for (String word : words) {
		String temp = trans(word);
		List<String> list;
		if (!map.containsKey(temp)) {
			list = new ArrayList<>();
			list.add(word);
			map.put(temp, list);
		} else {
			map.get(temp).add(word);
		}
	}
}
public String trans (String str) {
	Map<Character, List<Integer>> map = new HashMap<>();
	for (int i = 0; i < str.length(); i++) {
		char c = str.charAt(i);
		List<Integer> list;
		if (map.containsKey(c)) {
			map.get(c).add(i);
		} else {
			list = new ArrayList<>();
			list.add(i);
			map.put(c, list);
		}
	}
	int counter = 0;
	char[] chars = new char[str.length()];
	for (int i = 0; i < str.length(); i++) {
		if (!map.containsKey(str.charAt(i))) {
			continue;
		}
		List<Integer> list = map.get(str.charAt(i));
		map.remove(str.charAt(i));
		for (int j = 0; j < list.size(); j++) {
			char[j] = counter;
		}
		counter++;
	}
}



Given n sorted arrays, find first k common elements in them. 
E.g. the common elements of {{1,3,5,7}, {3,4,5,6}, {3,5,6,8,9}, {2,3,4,5,11,18}} are 3 and 5.


public int[] firstKCommonElements(int[][] arrays, int k) {
	int len = arrays.length;
	int[] pointers = new int[len];
	int[] result = new int[k];
	int index = 0;
	for (int i = 0; i < arrys[0].length; i++) {
		int pivot = arrays[0][i];
		int counter = 1;
		for (int j = 1; j < len; j++) {
			while (pointers[j] < arrays[j].length && pivot > arrays[j][pointers[j]]) {
				pointers[j]++;
			}
			if (pointers[j] == arrays[j].length || pivot != arrays[j][pointers[j]]) {
				break;
			}
			counter++;
		}
		if (counter == n) {
			result[index++] = pivot;
		}
		if (index == k) {
			return result;
		}
	}
	return result;
}






public int shorestPath (TreeNode root, TreeNode t1, TreeNode t2) {

	if (LCA(root, t1, t2) == root) {
		return getLen(root, t1) + getLen(root, t2);
	} else {
		if (LCA()) {

		}
	}

}

public TreeNode LCA (TreeNode root, TreeNode p, TreeNode q) {

}

public int getLen (TreeNode n1, TreeNode n2, int len) {
	if (n1 == null) {
		return 0;
	}
	if (n1 == n2) {
		return len;
	}
	int left = getLen(n1.left, n2, len + 1);
	int right = getLen(n2.right, n2, len + 1);

	return Math.max(left, right);
}



public String compressedString (String input) {
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








class Event {
	String winner;
	String loser;
	Event (Stirng winner, String loser) {
		this.winner = winner;
		this.loser = loser;
	}
}

public boolean hasCircle (Event[] events) {
	Map<String, Integer> indegree = new HashMap<>();
	Map<Stirng, List<String>> graph = new HashMap<>();
	for () {

	}
	Queue<Event> queue = new LinkedList<>();
	for (String key : indegree.keySet()) {
		if () {
			queue.offer();
		}
	}
}




public String reverse (String num) {
	char[] chars = num.toCharArray();
	helper (chars, 0, num.length());
	return new String(chars);
}
public void helper () {
	if (left >= right) {
		return;
	}
	char temp = chars[left];
	chars[left] = chars[right];
	chars[right] = temp;
	
	helper(chars, left + 1, right - 1);
}



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





public int reverseInt (int num) {
	//beyond the limitation
	if (num == 0 || num == Integer.MIN_VALUE) {
		return 0;
	}
	int result = 0;
	while (num != 0) {
		int temp = num % 10;
		if (num > 0 && result > (Integer.MAX_VALUE - temp) / 10 {
			return 0;
		}
		result = result * 10 + temp;
		num /= 10;
	}
	return num > 0 ? result : -result;
}




class User {
	public void generateRequest (int targetFloor) {

	}
}
class Elevator {
	int currentFloor;
	int targetFllor;

}
class RequestHandler {
	List<Request> reqs;
	RequestHandler () {
		this.reqs = new ArrayList<>();
	}
	void addRequest () {

	}

}
class Request {
	int requestFloor;
	Request instance = null;
	
	Request (int requestFloor) {
		this.requestFloor = requestFloor;
	}

	public int getTargetFloor () {

	}
}



public int sumUp (int n) {
	if (n == 1) {
		return n;
	} else if (n > 1) {
		sumUp()
	}
	return sumUp(n - 1);
}







int count = 0;
public int countUnivalSubtrees(TreeNode root) {

}
public boolean helper (TreeNode root) {
	if (root == null) {
		return true;
	}

	boolean left = helper(root.left);
	boolean right = helper(root.right);

	if (left && right) {

	}

	if (root.left != null && root.left.val != root.val) {
		return false;
	} else if (root.right != null && root.right.val != root.val) {
		return false;
	} else if (root.left == null && root.right == null) {
		count++;
		return true;
	} else {
		if (root.val != root.left.val || root.val != root.right.val) {
			return false;
		} else {
			count++;
			return true;
		}
	}

	
}




Given a binary search tree (BST) with duplicates, find all the mode(s) (the most frequently occurred element) 
in the given BST.

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than or equal to the node's key.
The right subtree of a node contains only nodes with keys greater than or equal to the node's key.
Both the left and right subtrees must also be binary search trees.
For example:
Given BST [1,null,2,2],
   1
    \
     2
    /
   2
return [2].

Note: If a tree has more than one mode, you can return them in any order.

Follow up: Could you do that without using any extra space? (Assume that the implicit stack space incurred due to 
recursion does not count).


int count = 1;
int max = 0;
Integer pre = null;
public int[] findMode(TreeNode root) {
	//iterate the tree, and record the occurance     
	if (root == null) {
		return new int[0];
	}
	List<Integer> list = new ArrayList<>();
	helper(list, root);

	return list.toArray();   
}
public void helper (List<Integer> list, TreeNode root) {
	if (root == null) {
		return;
	}
	helper(list, root.left);
	if (pre != null) {
		if (root.val == pre) {
			count++;
		} else {
			count = 1;
		}
	}
	if (count > max) {
		max = count;
		list.clear();
		list.add(root.val);
	} else if (count == max) {
		list.add(root.val);
	}
	pre = root.val;
	helper(list, root.right);

}









public List<List<Integer>> getFactors(int n) {
	List<List<Integer>> result = new ArrayList<>();
	List<Integer> list = new ArrayList<>();
	if (n <= 1) {
		return result;
	}
	result.add(new ArrayList<>(Arrays.asList(1, n)));
	helper(result, list, n, 2);
	return result;
}
public void helper (List<List<Integer>> result, List<Integer> list, int n, int index) {
	for (int i = index; i <= Math.sqrt(n); i++) {
		if (n % i == 0 && n / i >= i) {
			list.add(i);
			list.add(n / i);
			result.add(new ArrayList<>(list));
			list.remove(list.size() - 1);	
			helper(result, list, n / i, i);
			list.remove(list.size() - 1);
		}

	}
}






public List<String> wordsAbbreviation(List<String> dict) {

	int[] prefixNum = new int[dict.size()];
	String[] result = new String[dict.size()];
	for (int i = 0; i < dict.size(); i++) {
		result[i] = wordAbbre(dict.get(i), 1);
		prefixNum[i] = 1;
	}

	for (int i = 0; i < result.length; i++) {
		while (true) {
			Set<Integer> set = new HashSet<>();
			for (int j = i + 1; j < result.length; j++) {
				if (result[i].equals(result[j])) {
					set.add(j);
				}
			}
			if (set.isEmpty()) {
				break;
			}
			set.add(i);
			for (int pos : set) {
				result[pos] = wordAbbre(dict.get(pos), ++prefixNum[pos]);
			}
		}
	}
	return Arrays.asList(result);
}


public String wordAbbre (String str, int prefix) {
	if (str.length() - prefix <= prefix) {
		return str;
	}
	StringBuilder sb = new StringBuilder();
	sb.append(str.substirng(0, prefix));
	sb.append(str.length() - prefix - 1);
	sb.append(str.charAt(str.length() - 1));
	return sb.toString;
}

public class Solution {

    public int findLUSlength(String[] strs) {
    	Set<String> candidates = new HashSet<>();
    	Set<String> impossibles = new HashSet<>();
    	for (String str : strs) {
    		if (!impossibles.contains(str)) {
    			if (candidates.contains(str)) {
    				candidates.remove(str);
    				impossibles.add(str);
    			} else {
    				boolean isValid = true;
    				for (String imp : impossibles) {
    					if (isSubsequence(imp, str)) {
    						impossibles.add(str);
    						isValid = false;
    						break;
    					}
    				}
    				if (!isValid) {
 						candidates.remove(str);
 						impossibles.add(str);
    				} else { 
    					candidates.add(str);
    				}
    			}
    		}
    	}
    	int maxVal = -1;
    	for (String cand : candidates) {
    		maxVal = Math.max(maxVal, cand.length());
    	}
    	return maxVal;
    }

    public boolean isSubsequence (String pattern, String input) {
    	if (pattern.length() < input.length()) {
    		return false;
    	} else if (pattern.length() == input.length()) {
    		return pattern.equals(input);
    	} else {
    		int index = 0;
    		for (int i = 0; i < input.length(); i++) {
    			index = pattern.indexOf(input.charAt(i), index);
    			if (index == -1) {
    				return false;
    			}
    			index++;
    		}
    	}
    	return true;
    }
}





import java.io.*;
import java.util.*;

/* Imagine we have an image. We’ll represent this image as a simple 2D array where every pixel is a 1 or a 0. The image you get is known to have a single rectangle of 0s on a background of 1s. Write a function that takes in the image and returns the coordinates of the rectangle -- either top-left and bottom-right; or top-left, width, and height.

top left: row 2 col 3
bot right: row 3 col 5
 */

class Solution {
  public static void main(String[] args) {
    int[][] image = {
      {1, 0, 1, 1, 1, 1, 1},
      {1, 0, 0, 1, 0, 1, 1},
      {1, 1, 1, 0, 0, 0, 1},
      {1, 0, 1, 1, 0, 1, 1},
      {1, 0, 1, 1, 1, 1, 1},
      {1, 0, 0, 0, 0, 1, 1},
      {1, 1, 1, 0, 0, 1, 1},
      {1, 1, 1, 1, 1, 1, 1},
    };
    
    
    Solution sol = new Solution();
    List<List<Integer>> result = sol.findPos(image);
    
    for (List<Integer> list : result) 
      System.out.println(Arrays.toString(list.toArray()));
    
  }
  
  // top, left, right, bottom 
  public List<List<Integer>> findPos (int[][] matrix) {
    
    List<List<Integer>> result = new ArrayList<>();
    int rowNum = matrix.length, colNum = matrix[0].length;
    
    boolean[][] visited = new boolean[rowNum][colNum];
    
    for (int i = 0; i < rowNum; i++) {
      for (int j = 0; j < colNum; j++) {
        if (!visited[i][j] && matrix[i][j] == 0) {
          List<Integer> list = new ArrayList<>();
          list.add(i * matrix[0].length + j);
          helper(matrix, list, i, j, visited);
          result.add(new ArrayList<>(list));
        }
      }
    }
    return result;
  }
  
  public void helper (int[][] matrix, List<Integer> list, int row, int col, boolean[][] visited) {
    
    int[][] dirs = new int[][]{{0,1},{1,0},{-1,0},{0,-1}};
    
    visited[row][col] = true;
    
    for (int i = 0; i < dirs.length; i++) {
      int curRow = row + dirs[i][0];
      int curCol = col + dirs[i][1];
      
      // bounary
      if (0 > curRow || curRow >= matrix.length || 0 > curCol || curCol >= matrix[0].length) {
        continue;
      }
      // 0 < curRow ? 0 < -1
      // I checked above
      
      if (visited[curRow][curCol] || matrix[curRow][curCol] == 1) {
        continue;
      }
      
      // looks good -- rejoin?
      // can you hear me? no
      // still no
      list.add(curRow * matrix[0].length + curCol);
      
      helper(matrix, list, curRow, curCol, visited);
      
      
    }
  }
  

  
  
}









Map<String, Set<String>> IdToFriendsMap 
Map<String, Integer> result = new HashMap<>();

for (String id : IdToFriendsMap.keySet()) {
	Set<String> friends = IdToFriendsMap.get(id);
	for (String ff : friends) {
		if (!friends.contains(ff)) {
			if (!result.containsKey(ff)) {
				result.put(ff, 0);
			} 
			result.put(ff, result.get(ff) + 1);
		}
	}
}
return result;




String[] employeeInput = new String[]{"1,Alice,HR", "2,Bob,Engineer", "3,Daniel,Engineer","4,Chirley,Design","5,Bob,HR"};
String[] friendsInput = new String[]{"1,2","1,3","2,4"};

System.out.println("how many people in dep have other dep");
Map<String, Integer> result = otherDep(employeeInput, friendsInput);
for (String key : result.keySet()) {
	System.out.println(key);
	System.out.println(result.get(key));
}
System.out.println("friends list  ");
Map<String, Set<String>> friendsList = findFriend(employeeInput, friendsInput);
for (String key : friendsList.keySet()) {
	System.out.println("emp " + key);
	System.out.println(Arrays.toString(friendsList.get(key).toArray()));
}
System.out.println("isAllConnected ?  ");
System.out.println(sol.isAllConnected(employeeInput, friendsInput));
}


//frindList helper funciton
public static Map<String, Set<String>> IdToFriendsMap (String[] employeeInput, String[] friendsInput) {
	Map<String, Set<String>> map = new HashMap<>();

	for (String emp : employeeInput) {
		String[] temp = emp.split("\\s+");
		Set<String> set = new HashSet<>();
		map.put(temp[0], set);		
	}

	for (String friends : friendsInput) {
		String[] fri = friends.split("\\s+");			
		map.get(fri[0]).add(fri[1]);
		map.get(fri[1]).add(fri[0]);							
	}
	return map;
}

//找出每个人有多少好友 
//find friends list: [3: (1,5), 2: (1,4), 1: (3,2), 5: (3), 4: (2)]	
public static List<String> findFriendsList (String[] employeeInput, String[] friendsInput) {

	List<String> result = new ArrayList<>();
	Map<String, Set<String>> IdToFriendsMap = IdToFriendsMap(employeeInput, friendsInput);

	for (String emp : IdToFriendsMap.keySet()) {
		String temp = emp + ": (";

		if (IdToFriendsMap.get(emp).size() == 0) {
			temp += "null";
		} else {
			for (String e : IdToFriendsMap.get(emp)) {
				temp += e + ",";
			}
			temp = temp.substring(0, temp.length() - 1);
		}
		temp += ")";
		result.add(temp);
	}
	return result;
}






//find the number of employee in each department whoes friends are in other departments [Eng: 0 of 2, HR: 1 of 2, Design: 1 of 1]
public static List<String> depDetail (String[] employeeInput, String[] friendsInput) {
	List<String> result = new ArrayList<>();
	//fiend the friendList key : id, val, set of department
	Map<String, Set<String>> IdToFriendsMap = IdToFriendsMap(employeeInput, friendsInput);;	
	// key : id, val : department
	Map<String, String> idToDepMap = new HashMap<>();
	//key:department, val: list of id
	Map<String, List<String>> depToIdsMap = new HashMap<>();
	
	for (String emp : employeeInput) {
		String[] temp = emp.split("\\s+");
		//set idToDepMap
		idToDepMap.put(temp[0], temp[2]);
		// set depToIdMap
		List<String> list;
		if (!depToIdsMap.containsKey(temp[2])) {
			list = new ArrayList<>();
		} else {
			list = depToIdsMap.get(temp[2]);
		}
		list.add(temp[0]);
		depToIdsMap.put(temp[2], list);
	} 
	
	// iterate depToIdMap and find the number of employee whoes friends are in other departments
	for (String department :depToIdsMap.keySet()) {
		
		//employee whoes friends are in other dpeartments
		int count = 0;
		for (String employee : depToIdsMap.get(department)) {
			if (IdToFriendsMap.get(employee).size() == 0) {
				break;
			}
			for (String friend: IdToFriendsMap.get(employee)) {
				
				if (!idToDepMap.get(friend).equals(department)) {
					count++;
					break;
				}
			}				
		}
		String temp = department + ": " + count + " of " + String.valueOf(depToIdsMap.get(department).size());
		result.add(temp);
	}
	
	return result;
}





//输出是否所有employee都在一个社交圈
public static boolean isAllConnected (String[] employeeInput, String[] friendsInput) {

//start from a node and check if all empoyees can be visited;
Map<String, Set<String>> IdToFriendsMap = IdToFriendsMap(employeeInput, friendsInput);
Map<String, Boolean> visited = new HashMap<>();
String empId = "";
for (String key : IdToFriendsMap.keySet()) {
	if (empId.length() == 0) {
		empId = key;
	}
	visited.put(key, false);
}

helper(visited, empId, IdToFriendsMap);

for (String empolyee : visited.keySet()) {
	if (!visited.get(empolyee)) {
		return false;
	}
}

return true;
}

public static void helper (Map<String, Boolean> visited, String empId, Map<String, Set<String>> friendsMap) {
Set<String> friendsList = friendsMap.get(empId);
visited.put(empId, true);
		
for (String friendId : friendsList) {
	
	if (friendsMap.get(friendId).size() > 0 && !visited.get(friendId)) {
		visited.put(friendId, true);
		helper(visited, friendId, friendsMap);
	}
}
}	


public List<String> combinations (String input) {
	List<String> result = new ArrayList<>();
	helper(input, "", 0);
	return result;
}
public void helper (String input, String str, int index) {
	result.add(new String(str));
	for (int i = index; i < input.length(); i++) {
		helper(input, str + "" + input.charAt(i), i + 1);
	}
}


class TreeNode {
	TreeNode[] children;
	TreeNode node;
	TreeNode (TreeNode node, TreeNode[] children) {
		node = node;
		children = children;
	}
}

int maxDepth = 0;
public TreeNode lca (TreeNode root) {
	//get the deepest height left nodes;
	List<TreeNode> nodes = new ArrayList<>();
	TreeNode node = root;
	helper(node, nodes, 0);

}
public void helper (TreeNode node, List<TreeNode> nodes, int depth) {
	if (node == null) {
		return;
	}
	if (node.left == null && node.right == null) {
		if (depth > maxDepth) {
			nodes = new ArrayList<>();
			nodes.add(node);
		} else if (depth == maxDepth) {
			nodes.add(node);
		} 
		return;
	}
	for (TreeNode child : node.children) {
		helper(child, nodes, depth + 1);
	}
	
}


public List<List<String>> combinations (char[] chars) {
	List<List<String>> result = new ArrayList<>();
	for (int i = 1; i <= chars.length; i++) {
		List<String> list = new ArrayList<>();
		helper(list, chars, "", i, 0);
		result.add(list);
	}
	return result;
}
public void helper (List<String> list, char[] chars, String str, int k, int index) {
	if (str.length() == k) {
		list.add(new String(str));
		return;
	}
	for (int i = index; i < chars.length; i++) {
		helper(list, chars, str + chars[i] + "", k, i + 1);
	}
}




public int longestPalindrome (String input) {
	int max = 1;
	helper(input, 0);
	helper(input, 1);
	return max;
}
public void helper (String input, int diff) {
	for (int i = 0; i < input.length(); i++) {
		
		int left = i, right = i + diff;
		while (left >= 0 && right < input.length() && input.charAt(left) == input.charAt(right)) {
			left--;
			right++;
		}
		max = Math.max(right - left - 1, max);
	}
}


public int atoi (String input) {
	if (input == null || input.length() == 0) {
		return 0;
	}
	input.trim();
	int index = 0;
	// 0 means pos, 1 means neg
	int sign = 0;
	if (input.charAt(0) == '-') {
		sign = 1;
		index++;
	} 

	int num = 0;
	for (int i = index; i < input.length(); i++) {
		char c = input.charAt(i);
		if (!Character.isDigit(c)) {
			break;
		} else {
			if (sign == 0 && (num > Integer.MAX_VALUE / 10 
							  || (num == Integer.MAX_VALUE / 10 && (c - '0') > Integer.MAX_VALUE % 10)
							  )} {
				return Integer.MAX_VALUE;
			}
			if (sign == 1 && (num > Math.abs(Integer.MIN_VALUE / 10) || ) {
				return Integer.MIN_VALUE;
			}
			num = num * 10 + (c - '0');
		}
	}
	return sign == 0 ? num : -num;
}



public List<List<Integer>> combinations (int[] nums, int target, int k) {
	Arrays.sort(nums);
	List<List<Integer>> result = new ArrayList<>();
	List<Integer> list = new ArrayList<>();
	helper(nums, target, k, 0, 0, result, list);
	return result;
}
public void helper (int[] nums, int target, int k, int curSum, 
					int index, List<List<Integer>> result, List<Integer> list) {
	
	if (curSum == target && list.size() == k) {
		result.add(new ArrayList<>(list));
		return;
	}
	for (int i = index; i < nums.length; i++) {
		if (i != index && nums[i] == nums[i - 1]) {
			continue;
		}
		list.add(nums[i]);
		helper(nums, target, curSum + nums[i], i, result, list);
		list.remove(list.size() - 1);
	}
}


class node {
	int maxSum;
	int top;
	int bottom;
	int left;
	int right;
	node () {
		this.maxSum = Integer.MIN_VALUE;
		this.top = 0;
		this.bottom = 0;
		this.left = 0;
		this.right = 0;
	}
}
public node maxSubSumMatrix (int[][] matrix) {
	node result = new node();

	for (int i = 0; i < matrix[0].length; i++) {
		int[] curSum = new int[matrix.length];
		for (int j = i; j < matrix[0].length; j++) {
			for (int k = 0; k < matrix.length; k++) {
				curSum[k] = matrix[i][k];
			}
			int[] helper = maxSubSum(nums);
			if (helper[0] > node.maxSum) {
				result.maxSum = helper[0]
				result.left = i;
				result.right = j;
				result.top = helper[1];
				result.bottom = helper[2];
			}
		}
	}
	return result;
}

public int[] maxSubSum (int[] nums) {

	int maxVal = 0;
	int top = -1;
	int down = -1;
	int newStart = 0;
	int curSum = 0;

	for (int i = 0; i < nums.length; i++) {
		curSum += nums[i];
		if (curSum > maxVal) {
			maxVal = curSum;
			top = newStart;
			down = i;
		}
		if (curSum < 0) {
			curSum = 0;
			newStart = i + 1;
		}
	}
	//maxVal, top, down
	int[] result = new int[]{maxVal, top, down}
	return result;
}




public int maxSum (int[] nums) {
	int maxVal = Integer.MIN_VALUE;
	int curSum = 0;
	for (int num : nums) {
		curSum += num;
		maxVal = Math.max(curSum, maxVal);
		if (curSum < 0) {
			curSum = 0;
		}
	}
	return maxVal;
}

public void sortColor (int[] nums) {
	int left = 0, right = nums.length - 1;
	int i = 0;
	while (i < right) {
		if (nums[i] == 0) {
			swap(nums, left, i);
			left++;
			i++;
		} else if (nums[i] == 2) {
			swap(nums, right, i);
			right--;
		} else {
			i++;
		}
	}
}
public void swap (int[] nums, int i, int j) {
	int temp = nums[i];
	nums[i] = nums[j];
	nums[j] = temp;
}

public List<String> isTrue (String str1, String str2) {
	List<String> result = new ArrayList<>();
	StringBuilder sb = new StringBuilder();
	helper(result, 0, str1, 0, str2, sb);
	return result;
}
public void helper (List<String> result, int index1, String str1, int index2, String str2, StringBuilder sb) {
	
	if (index1 == str1.length() && index2 == str2.length()) {
		result.add(sb.toString);
		return;
	}
	if (index1 < str1.length()) 
		helper(index1 + 1, str1, index2, str2, sb);
	sb.deleteAt(sb.length() - 1);
	
	sb.append(str2.charAt(index2));
	helper(index1, str1, index2 + 1, str2, sb);
	sb.deleteAt(sb.length() - 1);
}

public int editString (String input, String target) {

	int[][] dp = new int[target.length() + 1][input.length() + 1];
	dp[0][0] = 0;
	for (int j = 0; j <= input.length(); j++) {
		dp[0][j] = j;
	}
	for (int i = 0; i <= target.length(); i++) {
		dp[i][0] = i;
	}
	int result = Integer.MAX_VALUE;
	for (int i = 1; i <= target.length(); i++) {
		for (int j = 1; j <= input.length(); j++) {
			if (input.charAt(j) == target.charAt(i)) {
				result = Math.min(dp[i - 1][j - 1], dp[i][j - 1] + 1, dp[i - 1][j] + 1)
			} else {
				dp[i - 1][j - 1] + 1, dp[i][j - 1] + 1, dp[i - 1][j] + 1 
			}
		}
	}
}


public List<Integer> maxConnectiveProduct (int[] nums) {
	List<Integer> result = new ArrayList<>();
	int len = nums.length;
	int[] max = new int[len];
	int[] min = new int[len];
	max[0] = min[0] = nums[0];
	
	int maxVal = nums[0];
	result.add(nums[0]);

	for (int i = 1; i < nums.length; i++) {
		if (nums[i] > 0) {
			if (max[i - 1] * nums[i] > nums[i]) {
				result.add(nums[i]);
			} else {
				result = new ArrayList<>();
			}
			max[i] = Math.max(nums[i], max[i - 1] * nums[i]);
			min[i] = Math.min(min[i - 1] * nums[i], nums[i]);
		} else if (nums[i] < 0) {
			max[i] = Math.max(nums[i] * min[i - 1], nums[i]);
			min[i] = Math.min(nums[i] * max[i - 1], nums[i]);
		} else {
			max[i] = min[i] = 0;
		}

	}
	return result;
}


public boolean binarySearch (int[] nums, int target) {
	int left = 0, right = nums.length - 1;
	while (left + 1 < right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] > target) {
			right = mid;
		} else if (nums[mid] < target) {
			left = mid;
		} else {
			return true;
		}
	}
	if (nums[left] == target) {
		return true;
	}
	if (nums[right] == target) {
		return true;
	}
	return false;
}


public boolean binarySearch (int[] nums, int target) {
	int left = 0, right = nums.length - 1;
	while (left <= right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] < target) {
			left = mid + 1;
		} else if (nums[mid] > target) {
			right = mid - 1;
		} else {
			return true;
		}
	}
	return false;
}


o(n) o(n)
nlg(n) o(1)
o(n) o(1)

int num = nums[0];
int count = 1;
for (int i = 1; i < nums.length; i++) {
	if (num == nums[i]) {
		count++;
	} else {
		count--;
		if (count == 0) {
			num = nums[i];
			count = 1;
		}
	}
}
if (count > 0) {
	return num;
}
return -1;
/*
刚刚面完JET的外包面试，不是找矩形的那道题了，不过也是面经里的一个.
是类似朋友圈的问题，一个公司给你两个String[] employeeInput = {"1,Alice,HR", "2,Bob,Engineer".....}, 
String[] friendsInput = {"1,2","1,3","2,4"}
1st Question: 输出所有的employee的friendlist -> 就是用一个map存起来然后打印就好了
（这个是无向图，e.g: 1和2是朋友，2的列表里也要有1）
2nd Question: 输出每个department里有多少人的朋友是其他部门的 ->也就是遍历一遍就好了
3rd Question: 输出是否所有employee都在一个社交圈 -> 我当时想的就是随便找一个点，用DFS遍历一遍，
如果所有点都被遍历到就return true，不然就是false
*/

//输出所有的employee的friendlist
public Map<String, Set<String>> findFriend (String[] employeeInput, String[] friendsInput) {

	Map<String, Set<String>> friendsMap = new HashMap<>();

	for (String emp : employeeInput) {
		String[] temp = emp.split(",");
		Set<String> set = new HashSet<>();
		friendsMap.put(temp[0], set);
	}

	for (String friends : friendsInput) {
		String[] fri = friends.split(",");
		if (!friendsMap.get(fri[0]).contains(fri[1])) {
			friendsMap.get(fri[0]).add(fri[1]);
		}
		if (!friendsMap.get(fri[1]).contains(fri[0])) {
			friendsMap.get(fri[1]).add(fri[0]);
		}

	}

	return friendsMap;

}

//输出每个department里有多少人的朋友是其他部门的
public Map<String, Integer> otherDep (String[] employeeInput, String[] friendsInput) {
	Map<String, Set<String>> friendsMap = findFriend(employeeInput, friendsInput);
	Map<String, Integer> result = new HashMap<>();

	Map<String, Set<String>> depFriendList = new HashMap<>();
	Map<String, String> empToDep = new HashMap<>();

	for (String emp : employeeInput) {
		String[] temp = emp.split(",");
		empToDep.put(temp[0], temp[2]);
		if (!depFriendList.containsKey(temp[2])) {
			Set<String> set = new HashSet<>();
			set.add(temp[0]);
			depFriendList.put(temp[2], set);
		} else {
			depFriendList.get(temp[2]).add(temp[0]);
		}
	}

	for (String dep : depFriendList.keySet()) {
		int num = 0;	
		for (String emp : depFriendList.get(dep)) {
			Set<String> frinds = frindsMap.get(emp);
			for (String friend : friends) {
				if (!empToDep.get(friend).equals(dep)) {
					num++;
					break;
				}
			}
		}
		result.add(dep, num);
	}	
	return result;
}

//输出是否所有employee都在一个社交圈
public boolean isAllConnected (String[] employeeInput, String[] friendsInput) {
	
	//start from a node and check if all empoyees can be visited;
	Map<String, boolean> visited = new HashMap<>();
	Map<String, Set<String>> friendsMap = new HashMap<>();

	for (String emp : employeeInput) {
		String[] temp = emp.split(",");
		friendsMap.put(temp[0], new HashSet<>());
		visited.put(temp[0], false);
	}
	//get the friendsList map
	for (String friends : friendsList) {
		String[] fri = friends.split(",");
		if (!friendsMap.get(fri[0]).contains(fri[1])) {
			friendsMap.get(fri[0]).add(fri[1]);
		}
		if (!friendsMap.get(fri[1]).contains(fri[0])) {
			friendsMap.get(fri[1]).add(fri[0]);
		}
	}

	String emp = employeeInput[0].split(",")[0];

	helper(visited, emp, friendsMap);

	for (String emp : visited.keySet()) {
		if (!visited.get(emp)) {
			return false
		}
	}

	return true;
}


public void helper (Map<String, boolean> visited, String emp, Map<String, Set<String>> friendsMap) {
	Set<String> friends = friendsMap.get(emp);
	visited.put(emp, true);
	for (String friend : friends) {
		if (visited.get(friend)) {
			return;
		} else {
			helper(visited, friend, friendsMap);
		}
	}
}










class Pos {
	int x;
	int y;
	Pos (int x, int y) {
		this.x = x;
		this.y = y;
	}
}
public List<List<Pos>> findPos (int[][] matrix) {

	List<List<Pos>> result = new ArrayList<>();
	int rowNum = matrix.length, colNum = matrix[0].length;
	boolean[][] visited = new boolean[rowNum][colNum];
	
	for (int i = 0; i < rowNum; i++) {
		for (int j = 0; j < colNum; j++) {
			if (!visited[i][j] && matrix[i][j] == 1) {
				List<Pos> list = new ArrayList<>();
				helper(matrix, result, list, i, j, rowNum, colNum, visited);
				result.add(new ArrayList<>(list));
			}
		}
	}
	return result;
}
public void helepr (int[][] matrix, List<List<Pos>> result, List<Pos> list, int i, int j, int rowNm, int colNum, boolean[][] visited) {
	if (i >= rowNum || j >= colNum || visited[i][j] || matrix[i][j] == 0) {
		return;
	}
	
	list.add(new Pos(i, j));
	visited[i][j] = true;

	int[][] dirs = new int[][]{{0,1},{1,0},{0,-1},{-1,0}};
	
	for (int k = 0; k < dirs.length; k++) {
		int row = i + dirs[k][0], col = j + dirs[k][1];
		helper(matrix, result, list, row, col, rowNum, colNum, viisted);
	}

}








public boolean isPalindrome (String input) {
	Map<Character, Integer> map = new HashMap<>();
	for (char c : input.toCharArray()) {
		if (map.containsKey(c)) {
			map.put(c, map.get(c) + 1);
		} else {
			map.put(c, 1);
		}
	}

	boolean odd = false;

	for (char key : map.keySet()) {
		if (map.get(key) % 2 != 0) {
			if (odd) {
				return false;
			} else {
				odd = true;
			}
		}
	}
	return true;
}



class Mouse {
	public mouse () {

	}
	public String findCheese (Place startingPoint) {
		//BFS
		if (startingPoint.isWall()) {
			return null;
		}
		if (startingPoint.isCheese()) {
			return null;
		}
		Queue<String> words = new LinkedList<>();
		Queue<Place> queue = new LinkedList<>();
		queue.offer(startingPoint);
		words.offer("");

		while (!queue.isEmpty()) {
			int size = queue.size();
			for (int i = 0; i < size;i ++) {
				Place cur = queue.poll();
				String word = words.poll();

				String temp = word;				
				Place n = cur.goNorth();
				if (!n.isWall()) {
					if (n.isCheese()) {
						return temp + "N"; 
					} 
					temp += "N";
					queue.offer(n);
					words.offer(temp);
				}

				String temp = word;				
				Place n = cur.goNorth();
				if (!n.isWall()) {
					if (n.isCheese()) {
						return temp + "N"; 
					} 
					temp += "N";
					queue.offer(n);
					words.offer(temp);
				}

				String temp = word;				
				Place n = cur.goNorth();
				if (!n.isWall()) {
					if (n.isCheese()) {
						return temp + "N"; 
					} 
					temp += "N";
					queue.offer(n);
					words.offer(temp);
				}

				String temp = word;				
				Place n = cur.goNorth();
				if (!n.isWall()) {
					if (n.isCheese()) {
						return temp + "N"; 
					} 
					temp += "N";
					queue.offer(n);
					words.offer(temp);
				}
				
			}
		}		
	}

}





//sorted
//heap, sort from smal to large, pop k nums nlgn
//
public int[] closestKValues (int[] nums, int k, int target) {

} 


public boolean buyCake (int[] nums, int target]) {
	int min = Integer.MAX_VALUE, max = 0;
	for (int num : nums) {
		min = Math.min(num, min);
		max = Math.max(max, num);
	}
	if (target < min) {
		return false;
	}
	int[] result = new 
	for (int i = min; i <= max; i++) {

		for (int j = 0; j < nums.length; j++) {
			if (nums[j] > i) {
				continue;
			}
			if (nums[j] == i) {
				result[i]++;
			} else {
				result[i] = result[result[i] - nums[j]] + 
			}
		}	
	}
	return result[max] != 0;
}


 public int smallestRect (List<int[]> points) {
	
	int minArea = Integer.MAX_VALUE;

	Set<String> set = new HashSet<>();
	
	for (int[] point : points) {
		String temp = point[0] + "*" + points[1];
		set.add(temp);
	}

	for (int i = 0; i < points.size(); i++) {
		for (int j = i + 1; j < points.size(); j++) {
	
			int[] leftU = points[i] 
			int[] rightD = points[j];

			if (set.contains(rightD[0] + "*" + leftU[1]) 
				&& set.contains(leftU[0] + "*" + rightD[1])) {
				minArea = Math.min(minArea, Math.abs(leftU[0] - rightD[0]) * Math.abs(leftU[1] - rightD[1]));
			}

		}
	}
	return minArea

}

// +／-
int result = 0;
public int targetSum (int[] nums, int target) {
	helper(nums, target, index, 0);
	return result;
}
public void helper (int[] nums, int target, int index, int curSum) {
	
	if (index == nums.length && curSum == target) {
		result++;
		return;
	}
	if (index >= nums.length) {
		return;
	}
	int temp = curSum + nums[index];
	helper(nums, target, index++, temp);
	temp = curSum - nums[index];
	helper(nums, target, index++, temp);
}



int maxLen = 0;
public int longestCosecutiveSubsequence (TreeNode root) {
		
	helper(root);

	return maxLen;
}

public int helper () {
	if (root == null) {
		return 0;
	}

	if (root.left == null && root.right == null) {
		return 1;
	}
	
	int left = helper(root.left);
	int right = helper(root.right);

	if (root.left != null && root.left.val == root.val + 1) {
		left = left + 1;
	} else {
		left = 1;
	}

	if (root.right != null && root.right.val == root.val + 1) {
		right = right + 1;
	} else {
		right = 1;
	}

	int result =  Math.max(left, right);
	maxLen = Math.max(maxLen, result);
	return result;
}





public int getNum (int row, int col) {
	if (row == 1) {

	}
	int[] dp = new int[row];
	dp[0] = 1;
	dp[row - 1] = 1;
	for (int i = 1; i < row - 1; i++) {
		dp[i][] = dp[][]
	}
}

public List<TreeNode> nodePairs (TreeNode root) {
	List<List<TreeNode>> result = new ArrayList<>();
	if (root == null) {
		return result;
	}
	helper(root, result);
}
public void helper (TreeNode root, List<List<TreeNode>> result) {
	if (root == null) {
		return;
	}
	compare(root.left, root.right, result);
	helper(root.left, result);
	helper(root.right, result);
	
}
public void compare (TreeNode left, TreeNode right, List<List<TreeNode>> result) {
	if (left == null || right == null) {
		return;
	}
	if (isSame(left, right)) {
		result.add(new ArrayList<TreeNode>(Arrays.asList(left, right)));
	}
	compare(left, right.left, result);
	compare (left, right.right, result);
	compare(left.left, right, result);
	compare(left.right, right, result);
}

public boolean isSame (TreeNode r1, TreeNode r2) {
	if (r1 == null && r2 == null) {
		return true;
	} else if (r1 == null || r2 == null) {
		return false;
	} else {
		if (r1.val != r2.val) {
			return false;
		}
	}
	return helper(r1.left, r2.left) && helper(r1.right, r2.right);
}




public List<List<Integer>> getFactors(int n) {
         List<List<Integer>> result = new ArrayList<List<Integer>>();
         if (n <= 1) {
             return result;
         }
         List<Integer> list = new ArrayList<>();
         helper(result, list, n, 2);
         return result;
    }
    public void helper (List<List<Integer>> result, List<Integer> list, int n, int index) {
        
        for (int i = index; i <= (int)Math.sqrt(n) ; i++) {
            if (n % i == 0 && n / i >= i) {
                list.add(i);
                list.add(n / i);
                result.add(new ArrayList<>(list));
                list.remove(list.size() - 1);
                helper(result, list, n / i, i);
                list.remove(list.size() - 1);
            }
        }
        
    }


public TreeNode arrToBST (int[] nums) {
	if (nums == null) {
		return null;
	}
	return helper(nums, 0, nums.length - 1);
}

public TreeNode helper (int[] nums, int start, int end) {
	if (start > end) {
		return null;
	}
	int mid = start + (end - start) / 2;
	TreeNode root = new TreeNode(nums[mid]);
	root.left = helper(nums, start, mid - 1);
	root.right = helper(mid + 1, end);
	return root;
}

1->2->3 5->6->7

	3
  1
    

get tow point, then find if exist other tow points, get maxArea


public TreeNode linkedListToBST(ListNode root) {
	if (root == null) {
		return null;
	}
	ListNode node = mid(root);
	TreeNode result = new TreeNode(node.val);
	result.right = mid(node.next)
	node = null;
	result.left = mid(root);

}    

public ListNode mid (ListNode root) {
	if (root == null) {
		return null;
	}
	ListNode slow = root;
	ListNode fast = root;
	while (fast.next != null && fast.next.next != null) {
		fast = fast.next.next;
		slow = slow.next;
	}
	return slow;
}

public int getNum (String str) {
	Stack<String> sym = new Stack<>();
	Stack<Integer> nums = new Stack<>();
	Stack<Integer> count = new Stack<>();
	String[] input = str.split(" ");
	int counter = 0;
	for (int i = 0; i < input.length; i++) {
		String cur = input[i];
		if (cur.equals("(")) {
			if (counter > 0) {
				count.push(counter);
				counter = 0;
			}
			
		} else if (cur.equals(")")) {
			int num = 0;
			if (counter > 0) {
				String symbol = sym.pop();
				num = symbol.equals("*") ? 1 : 0;
				while (counter > 0) {
					if (symbol.equals("*")) {
						num = num * nums.pop();
					} else {
						num += nums.pop();
					}
					counter--;
				}
				nums.push(num);
				counter = nums.pop() + 1;
			}
		} else if (c == '*' || c == '+') {
			sym.push(c);
		} else {
			nums.push(Integer.parseInt(cur));
			counter++;
		}
	}
}






public List<String> longestPalindrome (String input) {
	List<String> result = new ArrayList<>();
	if (input == null || input.length() == 0) {
		return result;
	}
	Map<character, Integer> map = new HashMap<>();
	for (char c : input.toCharArray()) {
		if (!map.containsKey(c)) {
			map.put(c, 1);
		} else {
			map.put(c, map.get(c) + 1);
		}
	}

	List<Character> chars = new ArrayList<>();
	List<Integer> count = new ArrayList<>();

	char odd = '';
	int oddNum = 0;

	for (char key : map.keySet()) {
		if (key % 2 == 0) {
			chars.add(key);
			count.add(map.get(key));
		} else {
			if (map.get(key) > oddNum) {
				oddNum = map.get(key);
				odd = key;
			}
		}
	}

	StringBuilder sb = new StringBuilder();
	if (oddNum != 0) {
		sb.append(odd);
		map.put(odd, map.get(odd) - 1);
		if (map.get(odd) == 0) {
			map.remove(odd);
		}
	}
	if (chars.size() == 0) {
		result.add(sb.toString());
		return result;
	}
	String str = "";
	for (int i = 0; i < chars.size(); i++) {
		for (int i = 0; i < count.get(i) / 2; i++) {
			str += chars.get(i);
		}
	}
	// aabb
	List<String> list = new ArrayList<>();
	boolean[] visited = new boolean[str.length()];

	helper(str, list, visited, 0);
	for (String str : list) {
		String reverse = reverse(str);
		result.add(str + reverse + "");
	}
	return result;
}
public void helper (String input, List<String> list, boolean[] visited, int index) {
	for (int i = 0; i < input.length(); i++) {

	}
}
public String reverse (String input) {

}



public boolean isConnected (int[][] grid) {
	//dfs
	int count = 0;
	for (int i = 0; i < rowNum; i++) {
		for (int j = 0; j < colNum; j++) {
			if (gird[i][j] == 0) {
				if (count == 1) {
					return false;
				}
				helper(grid, i, j , count++);
			}
			
		}
	}
	return count <= 1;
	//bfs
}
public void helper (int[][] grid, int i, int j, int count) {
	
	int[][] dir = new int{{0,1}, {1,0},{0,-1},{-1,0}};

	if (i < 0 || i >= gird.length || j < 0 || j >= grid[0].length) {
		return;
	}

	grid[i][j] = 1;
	
	for (int i = 0; i < dir.length; i++) {
		int row = i + dir[i][0], col = j + dir[i][1];
		helper(gird, row, col, count); 
	}

}


//union found
int[] nums = new int[rowNum * colNum];

for (int i = 0; i < rowNum; i++) {
	for (int j = 0; j < colNum; j++) {
		if (grid[i][j] == 0) {
			nums[i * colNum + j] = 1; 
		}
	}
}




public int numofBlock (List<ListNode> pointers, DLLNode root) {
	
	Set<ListNode> set = new Hashset<>();
	for (ListNode pointer : pointers) {
		set.add(pointer);
	}
	int count = 0;

	while (!set.isEmpty()) {
		ListNode cur = set.iterator().next();
		ListNode node = cur.left;
		//left
		while (node != null) {
			if (set.contains(node)) {
				set.remove(node);
				node = node.left;
			} else {
				break;
			}
		}
		//right
		node = cur.right;
		while (node != null) {
			if (set.contains(node)) {
				set.remove(node);
				node = node.right;
			} else {
				break;
			}
		}
		count++;
	}
	return count;

} 





public void quickSort(int[] array) {
    if (array == null) {
      return;
    }
    quickSort(array, 0, array.length - 1);
  }

  public void quickSort(int[] array, int left, int right) {
    if (left >= right) {
      return;
    }
    // define a pivot and use the pivot to partition the array.
    int pivotPos = partition(array, left, right);
    quickSort(array, left, pivotPos - 1);
    quickSort(array, pivotPos + 1, right);
  }

private int partition(int[] array, int left, int right) {
	int pivotIndex = pivotIndex(left, right);
	int pivot = array[pivotIndex];
	// swap the pivot element to the rightmost position first
	swap(array, pivotIndex, right);
	int leftBound = left;
	int rightBound = right - 1;
	while (leftBound <= rightBound) {
	  if (array[leftBound] < pivot) {
	    leftBound++;
	  } else if (array[rightBound] >= pivot) {
		rightBound--;
	  } else {
	    swap(array, leftBound++, rightBound--);
	  }
	}
	// swap back the pivot element.
	swap(array, leftBound, right);
	return leftBound;
}


private int pivotIndex(int left, int right) {
	// sample implementation, pick random element as pivot each time.
	return left + (int) (Math.random() * (right - left + 1));
}

private void swap(int[] array, int left, int right) {
	int temp = array[left];
	array[left] = array[right];
	array[right] = temp;
}








public int findKthLargest(int[] nums, int k) {
        
	if (nums == null || nums.length == 0 || k <= 0 || k > nums.length) {
		return -1;
	}
	return helper(nums, 0, nums.length - 1, k - 1);
	
}
public int helper (int[] nums, int start, int end, int k) {
	int orgL = start, orgR = end;
	int mid = start + (end - start) / 2;

	while (start <= end) {		
		if (nums[start] > nums[mid]) {
			start++;
		} else if (nums[end] < nums[mid]) {
			end--;
		} else {
			int temp = nums[start];
			nums[start] = nums[end];
			nums[end] = temp;
			start++;
			end--;
		}
	}
	if (orgL < end && k <= end) {
		return helper(nums, orgL, end, k);
	} 
	if (start < orgR && k >= start) {
		return helper(nums, start, orgR, k);
	}
	return nums[k];
}






public boolean isSame (TreeNode r1, TreeNode r2) {
	if (r1 == null && r2 == null) {
		return true;
	} else if (r1 == null || r2 == null) {
		return false;
	} else if (r1.val != r2.val) {
		return false;
	}
	return isSame(r1.left, r2.left) && isSame(r1.right, r2.right);
}


public boolean isSubtree (TreeNode root, TreeNode sub) {
	if (sub == null) {
		return true;
	}
	if (root == null) {
		return false;
	}
	if (isSame(root, sub)) {
		return true;
	}

	return isSubtree(root.left, sub) || isSubtree(root.right, sub);
}




//bold font

class Interval {
	int start;
	int end;
	Interval (int start, int end) {
		this.start = start;
		this.end = end;
	}
}

public String boldFont (String input, List<String> sub) {
	List<Interval> intervals = new ArrayList<>();
	for (String str : sub) {
		int len = str.length();
		for (int i = 0; i <= input.length() - len; i++) {
			if (input.substring(i, i + len).equals(str);) {
				intervals.add(new Interval(i, i + len - 1));
			}
		}
	}
	if (intervals.size() == 0) {
		return input;
	}
	Collections.sort(intervals, new Comparator<Interval> (){
		public int compare (Interval i1, Interval i2) {
			return i1.start - i2.start;
		}
	});

	Interval pre = intervals.get(0);
	List<Interval> result = new ArrayList<>();
	for (int i = 1; i < intervals.size(); i++) {
		Interval cur = intervals.get(i);
		if (cur.start <= pre.end + 1) {
			pre.start = Math.min(pre.start, cur.start);
			pre.end = Math.max(pre.end, cur.end);
		}
		result.add(pre);
	}
	StringBuilder sb = new StringBuilder();
	int index = 0;
	for (Interval interval : result) {
		int start = interval.start;
		int end = interval.end;
		while (index < start) {
			sb.append(input.charAt(index);
			index++;
		}
		sb.append("<b>");
		while (index < end) {
			sb.append(input.charAt(index);
			index++;
		}
		sb.append("</b>");
	}
	while (index < input.length()) {
		sb.append(input.charAt(index++));
	}
	//merge intervals
	return sb.toString();
}




class Node {
	int key;
	int val;
	long exp;
	Node (int key, int val, long exp) {
		this.key = key;
		this.val = val;
		this.exp = exp;
	}
}

class ExperationMap {
	Map<Integer, Node> map;
	TreeMap<Long, Set<Integer>> expMap;
	ExperationMap () {
		this.map = new HashMap<>();
		this.expMap = new TreeMap<>();
	}

	public Integer get(int key) {
		if (!map.containsKey(key)) {
			return null;
		} else {
			Node cur = map.get(key);
			if (cur.exp < curTime) {
				map.remove(key);
				Set<Integer> list = expMap.get(cur.exp);
				list.remove(key);
				return null;
			}
			return map.get(key).val;
		}
	}
	public void set (int key, int val, long exp) {
		if (!map.containsKey(key)) {
			map.put(key, new Node(key, val, exp));
			Set<Integer> temp;
			if (!expMap.containsKey(key)) {
				temp = new HashSet<>();
			} else {
				temp = expMap.get(exp);
			}
			temp.add(key);
			expMap.put(exp, temp);
		} else {
			Node preNode = map.get(key);
			map.get(key).val = val;
			map.get(key).exp = exp;

			expMap.get(preNode.exp).remove(preNode.key);
			
			Set<Integer> temp;
			if (!expMap.containsKey(key)) {
				temp = new HashSet<>();
			} else {
				temp = expMap.get(exp);
			}
			temp.add(key);
			expMap.put(exp, temp);
		
		}
	}
}


public void clean (long exp) {
	List<Long> list = expMap.lower(exp);
	if (list.size() == 0) {
		return;
	}
	for (long expTime : list) {
		//remove in map
		Set<Integer> keys = expMap.get(expTime);
		for (int key : keys) {
			if (map.containsKey(key)) {
				map.remove(key);
			}
		}
		//remove in the expMap
		expMap.remove(expTime);
	}

}


public void printJson (String input) {

	Stack<Integer> pos = new Stack<>();
	int startIndex = 0, tab = 10;
	boolean newLine = true;

	StringBuilder temp = new StringBuilder();

	for (int i = 0; i < input.length(); i++) {
		char c = input.charAt(i);

		if (c == '{') {
			if (temp.length() > 0) {
				for (int i = 0; i < pos.peek(); i++) {
					System.out.println(" ");
				}
				System.out.prinltn(temp.toString());
				temp = new StringBuilder();
				System.out.println("\n");
				startIndex += tab;
			}

			if (newLine) {
				System.out.println("{");
				System.out.println("\n");
				pos.push(startIndex);
				startIndex += tab;
			} else {
				StringBuilder sb = new StringBuilder();
				while (input.charAt(i) != '}') {
					sb.append(input.charAt(i));
					i++;
				}
				sb.append(input.charAt(i));
				i++;
				if (input.charAt(i) == ',') {
					sb.append(",");
				}
				
				System.out.println("\n");
				for (int i = 0; i < pos.peek() + tab; i++) {
					System.out.println(" ");
				}
				System.out.println(sb.toString());

				i--;
			}
		} else if (c == '[') {
			if (temp.length() > 0) {
				for (int i = 0; i < pos.peek() + tab; i++) {
					System.out.println(" ");
				}
				System.out.prinltn(temp.toString());
				temp = new StringBuilder();
				System.out.println("\n");
				stack.pus
				startIndex += tab;
			}

			System.out.println("\n");
			stack.push(startIndex + tab);
			for (int i = 0; i < pos.peek() + tab; i++) {
				System.out.println(" ");
			}
			
		} else if (c == ']' || c == '}') {
			System.out.println(c);
			if (index + 1 < input.length() && input.charAt(++i) == ',') {
				System.out.println(",");
			}
			stack.pop();

		} else if (c == ',') {
			if (temp.length() > 0) {
				for (int i = 0; i < pos.peek() + tab; i++) {
					System.out.println(" ");
				}
				System.out.prinltn(temp.toString());
				System.out.println(",");
				temp = new StringBuilder();
			}
		} else {
			temp.append(input.charAt(i));
		}
	}

}





class RLEIterator implements Iterator{
	Iterator<Integer> values;
	RLEIterator(Iterator<Integer> input) {
		values = input;
	}
	
	public boolean hasNext() {
		return values.hasNext();
	}
	public Iterator<Integer> next() {
		int times = values.next();
		int num = values.next();

	}
}




class ListNode {
	int val;
	ListNode next;
	ListNode (int val, ListNode next) {
		this.val = val;
		this.next = null;
	}

}
public ListNode addOne (ListNode head) {
	if (head == null) {
		return;
	}
	//find the last one less than 9
	ListNode dummy = new ListNode(1);
	dummy.next = head;
	ListNode node = dummy;
	ListNode first = null, second = null;
	while (node.next != null) {
		if (node.val < 9) {
			first = node;
		}
		node = node.next;
	}

	if (node.val < 9) {
		node.val = node.val + 1;
		return dummy.next;
	} else {
		//999
		if (first == null) {
			node = dummy.next;
			while (node != null) {
				node.val = 0;
				node = node.next;
			}
			return dummy;
		//789	
		else {
			node = first;
			while (node != null) {
				if (node == first) {
					node.val = node.val + 1;
				} else {
					node.val = 0;
				}				
				node = node.next;
			}
		}
	}

}


public int daystoFlow (int[][] matrix, int start, int dest) {
	if (matrix == null || matrix.length || matrix[0].length || start == dest) {
		return 0;
	}

	int rowNum = matrix.length, colNum = matrix[0].length;
	int startDays = matrix[start / colNum][start % colNum];
	int result = startDays;
	
	Set<Integer> visited = new HashSet<>();
	
	Deque<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
		public int compare (Integer i1, Integer i2) {
			return matrix[i1 / colNum][i1 % colNum] - matrix[i2 / colNum][i2 % colNum];
		}
	});

	queue.add(start);
	visited.add(start);

	int[][] dir = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
	
	while (!queue.isEmpty()) {
		
		while (!queue.isEmpty() && matrix[queue.peek() / colNum][queue.peek() % colNum] < result) {
			int curHeight = queue.poll();
			
			for (int i = 0; i < 4; i++) {
				int curRow = curHeight / colNum + dir[i][0];
				int curCol = curHeight % colNum + dir[i][1];
				if (curRow * colNum + curCol == dest) {
					return result + 1;
				}
				if (visited.contains(curRow * colNum + curCol)) {
					continue;
				}
				if (matrix[curRow][curCol] <= result) {
					visited.add(curRow * colNum + curCol);
				}
				queue.offer(curRow * colNum + curCol);
			}

		}
		
		result++;
	}

	return result;
}

public int unionNode (List<Node> input) {
	Set<Node> nodes = new HashSet<>();
	for (Node n : input) {
		nodes.add(n);
	}
	int result = 0;
	for (Node cur : input) {
		//right
		Node temp = cur;
		while (nodes.contains(temp)) {
			nodes.remove(temp);
			temp = temp.next;
		}
		//left
		temp = cur.pre;
		while (nodes.contains(temp)) {
			nodes.remove(temp);
			temp = temp.pre;
		}
		result++;
	}
	return result;
}




TreeNode head = root;
while (head != null) {
	TreeNode next = new TreeNode(0);
	TreeNode cur = next;

	while (head != null) {
		if (head.left != null) {
			if (next.next == null) {
				next.next = head.left;
			}
			cur.next = head.left;
			cur = cur.next;
		}
		if (head.right != null) {
			if (next.next == null) {
				next.next = head.right;
			}
			cur.next = cur.right;
			cur = cur.next;
		}

	}
	head = next.next;
}

start from zero, if all around are zero skip, if there is one, count, if dist[i][j] < curDist, skip

List<Integer> visited = new ArrayList<>();


List<Domino> arrs = new ArrayList<>();

int preVal = Math.max(arrs[0].top, arrs[0].bottom);
for (int i = 1; i < arrs.length; i++) {

	int left = Math.max(arrs[i].top, arrs[i].bottom);
	int right = Math.min(arrs[i].top, arrs[i].bottom);

	if (visited.contains(right)) {
		return true;
	}
	visited.add(left);
	visited.add(right);

	if (left != preVal) {

	}


}




INNER JOIN:
SELECT Prcies.*, Quantity.QUANTITY
FROM Prcies INNER JOIN Quantities
ON Prcies.Product == Quantity.Product;

LEFT OUTER JOIN(右边包括左边出现过的，设置为null):
SELECT Prcies.*, Quantity.QUANTITY
FROM Prices LEFT OUTER JOIN Quantities
ON Prices.Product = Quantities.Product;



boolean search helper (TreeNode root, String word, boolean diff, int index){

	if (root.word != null) {
		if (diff && root.word.size() == word.length()) {
			return true;
		} else if (!diff && word.length() == root.word.size() + 1) {
			return true;
		}
	}


	if (!diff) {
		if (root.children[word.charAt(index) - 'a'] == null) {
			//replace
			boolean result = false;
			for (int i = 0; i < root.children.size(); i++) {
				if (root.children[i] == null) {
					continue;
				}
				result |=  helper(root.children[i], word, true, index + 1);
			}
			// goes to next index of word
			if (!result) {
				return helper(root, word, true, index + 1);
			} else {
				return true;
			}
		} else {
			return helper(root.children[word.charAt(index) - 'a'], word, diff, index + 1);
		}
	} else {
		if (root.children[word.charAt(index) - 'a'] == null) {
			return false;
		} else {
			return helper(root.children[word.charAt(index) - 'a'], word, diff, index + 1);
		}
	}
	return false;
}





//find Nth smallest in bst
public int findNthSmallest (Node root, int n) {
	
	if (root == null) {
		return -1;
	}
	if (getHeight(root) == n) {
		return root.val;
	} else if (getHeight(root) < n) {
		return findNthSmallest(root.left, n);
	} else {
		return findNthSmallest(root.right, n - getHeight(root));
	}
}
public int getHeight (Node root) {
	if (root == null) {
		return 0;
	}
	return getHeight(root.left) + getHeight(root.right) + 1;
}


//preorder get nth


// first find the two nodes, and then find first, middle, last three 
// if middle == t2, 
// else 


public void swap (Node node) {
	Node head = node;
	Node large = null, small = null;
	while (head != null) {
		if (large == null && small == null) {
			large = head;
		} else {

			if (head.val >= large) {
				small = large;
				large = head;
			} else {
				if (small == null) {
					small = head;
				} else {
					if (small.val < head.val) {
						small = head;
					}
				}
				
			}
		}
		head = head.next;
	}

	if (large == null || small == null) {
		return;
	}
	Node t1 = null, t2 = null, pre1 = null, pre2 = null;
	while (head.next != null) {
		if (head.next == large || head.next == small) {
			if (t1 == null) {
				pre1 = head;
				t1 = head.next;
			} else if (t2 == null) {
				pre2 = head;
				t2 = head.next;
			} else {
				break;
			}
		}
		head = head.next;
	}

	if (t1.next == t2) {
		pre1.next = t2;
		t1.next = t2.next;
		t2.next = t1;
	} else {
		pre1.next = t2;
		Node temp = t2.next;
		t2.next = t1.next;
		pre2.next = t1;
		t1.next = temp;
	}


}






public boolean fillWater(int num1, int num2, int target) {
	Set<Integer> visited = new HashSet<>();

	int t1 = num1, t2 = 0;
	while (!set.contains(t1)) {
		if (target == t1) {
			return true;
		}
		visited.add(t1);
		if (t2 + t1 < num2) {
			t2 += t1;
			if (t2 == target) {
				return true;
			}
			if (visited.contains(t2)) {
				return fale;
			}
			visited.add(t2);
		} else {
			t1 = t1 + t2 - num2;
			t2 = 0;
			if (t1 == target) {
				return true;
			}
		}
	}
	return false;
}




Map<String, List<String>> graph = new HashMap<>();
Map<String, Integer> indegree = new HashMap<>();

public void addRule(String name, List<String> dependencies) {
	if (!graph.containsKey(name)) {
		graph.put(name, dependencies);
	} else {
		graph.get(name).addAll(dependencies);
	}

	for (String dep : dependencies) {
		if (!indegree.containsKey(dep)) {
			indegree.get(dep)++;
		} else {
			indegree.put(dep, 1);
		}
	}
	
}
public List<String> getRules () {
	Queue<String> queue = new LinkedList<>();
	for (String key : indegree.keySet()) {
		if (indegree.get(key) == 0) {
			queue.offer(key);
		}
	}
	List<String> result = new ArrayList<>();
	while (!queue.isEmpty()) {
		int size = queue.size();
		for (int i = 0; i < size; i++) {
			String curStr = queue.poll();
			result.add(0, curStr);
			for (String next : graph.get(curStr)) {
				if (indegree.get(next) == 1) {
					queue.offer(next);
					indegree.put(next, 0);
				} else {
					indegree.put(next, indegree.get(next) - 1);
				}
			}
		}
	}

}


class ABC implements Runnable {
	private static int state = 0;
	
	public static void main (String[] args) {
		ABC sol = new ABC();
		Thread A = new Thread (new Runnable() {
			public void run () {
				for (int i = 0; i < 10; i++) {
					synchronized(sol) {
						while (state % 3 != 0) {
							try {
								sol.wait();
							} catch (InterruptedException e) {
								e.printStackTrace();
							}
						}

						System.out.println("A");
						state++;
						sol.notifyAll();
					}
				}
			}
		});

		Thread B = new Thread (new Runnable() {
			public void run () {
				for (int i = 0; i < 10; i++) {
					synchronized (sol) {
						while (state % 3 != 1) {
							try {
								sol.wait();
							} catch (InterruptedException) {
								e.printStackTrace();
							}
						}
						System.out.println("B");
						state++;
						sol.notifyAll();
					}
				}
			}
		});

		Thread C = new Thread (new Runnable() {
			public void run () {
				for (int i = 0; i < 10; i++) {
					synchronized (sol) {
						while (state % 3 != 2) {
							try {
								sol.wait();
							} catch (InterruptedException) {
								e.printStackTrace();
							}
						}
						System.out.println("C");
						state++;
						sol.notifyAll();
					}
				}
			}
		});
		
		A.start();
		B.start();
		C.start();

	}

}














int result = 0;
public int numOfdicount (String p, String str) {
	Map<Character, List<Integer>> map = new HashMap<>();
	for (int i = 0; i < str.length(); i++) {
		List<Integer> list;
		if (!map.containsKey(str.charAt(i))) {
			list = new ArrayList<>();
			map.put(str.charAt(i), list;
		} else {
			list = map.get(str.charAt(i));
		}
		list.add(i);
	}

	helper(p, str, 0, -1);
	return result;
}
public void helper (String p, String str, int index, int preVal) {
	if (index == p.length()) {
		result++;
		return;
	}
	char c = p.charAt(index);
	List<Integer> list = map.get(c);


	for (int i = 0; i < list.size(); i++) {
		if (preVal == -1) {
			helper(p, str, index + 1, list.get(i));
		} else if (preVal + 2 <= list.get(i)) {
			helper(p, str, index + 1, list.get(i));
		}
		
	}
	

}




int num = 0;
public List<Integer> numOfComb (int[] nums, int target) {
	List<Integer> result = new ArrayList<>();
	helper(nums, target, 0, 0, result, "");
	return result;
}
public void helper (int[] nums, int target, int index, int curSum, List<Integer> result, String path) {
	if (index == nums.length - 1 && curSum == target) {
		result.add(new String(path));
		return;
	}
		helper(nums, target, i + 1, curSum + nums[i], result, path + "+" + nums[i]);
		helper(nums, target, i + 1, curSum - nums[i], result, path + "-" + nums[i]);
	
}


class Map<K,V> {
	Map<K, V> map;
	Map<K, V> history;

	Map () {
		this.map = new HashMap<>();	
		this.history = new HashMap<>();
	}

	public K get (K key, V time) {

		if (!map.containsKey(key)) {
			return null;
		}
		if (time > history.get(key)) {
			return null;
		}
		return map.get(key);
	}
	
	public void set (K key, V val, V time) {
		
		map.put(key, val);
		history.put(key, time);

	}
	
	public void clean (V time) {
		Iterator<Map.Entry<K, V>> iter = history.entrySet().iterator();
		while (iter.hasNext()) {
		    Map.Entry<K, V> entry = iter.next();
		    if(entry.getKey() > time){
		        iter.remove();
		    }
		}
	}

}






//bfs 
public int max (int[] nums, int lower, int higher) {

}


public int[] recover (int[] nums) {
	Set<>
}

public int findMaxForm (int m, int n, String[] strs) {

	int[][] dp = new int[m + 1][n + 1];

	for (String s : strs) {
		int[] cost = count(s);
		for (int i = m; i >= cost[0]; i--) {
			for (int j = n; j >= cost[1]; j--) {
				dp[i][j] = Math.max(dp[i][j], dp[i - cost[0]][j - cost[1]] + 1);
			}
		}
	}
	return dp[m][n];
}
	
public int[] count (String str) {
	int[] cost = new int[2];
	for (int i = 0; i < str.length(); i++) {
		cost[str.charAt(i) - '0']++;
	}
	return cost;
}

//m表示0的个数，n表示1的个数

int maxCount = 0;
public int findMax (int m, int n, String[] inputs) {
	Arrays.sort(inputs, new Comparator<String>() {
		public int comoare (String str1, String str2) {
			return str1.length() - str2.length();
		}
	});

	helper(m, n, inputs, 0, 0, 0);
	
	return maxCount;
}
public void helper (int m, int n, String[] inputs, int count, int totalLen, int index) {
	
	if (totalLen > m + n) {
		return;
	}
	for (int i = index; i < inputs.length; i++) {
		int zero = 0;
		int ones = 0;
		for (int i = 0; i < inputs[i].length; i++) {
			char c = inputs[i].charAt(i);
			if (c == '1') {
				ones++;
			} else {
				zero++;
			}
		}
		if (m - zero < 0 || n - ones < 0) {
			continue;
		}
		maxCount = Math.max(maxCount, count + 1);
		helper(m - zero, n - ones, inputs, count + 1, totalLen + inputs[i].length(), i + 1);
	}
}


public List<String> combinations (int[] buttons) {

	List<String> result = new ArrayList<>();
	if (buttons == null || buttons.length == 0) {
		return result;
	}

	for (int i = 1; i <= buttons.length; i++) {
		Set<Integer> visited = new HashSet<>();
		helper(visited, result, buttons, new StringBuilder(), i, 0, 0);
	}
	return result;
}
public void helper (Set<Integer> visited, List<String> result, int[] buttons, StringBuilder sb, int totalTime, int index, int curTime) {
	if (curTime + 1 == totalTime) {
		if (index >= buttons.length - 1) {
			return;
		}
		String input = ""
		for (int i = 0; i < buttons.length; i++) {
			if (!visited.contains(buttons[i])) {
				input += buttons[i] + "";
			}
		}
		List<String> combi = combi (input);
		if (combi.size() == 0) {
			result.add(new String(sb.toString());
			return;
		}
		for (String com : combi) {
			result.add(sb.toString() + com);
		}
		
		return;
	}
	for (int i = 0; i <= buttons.length; i++) {
		if (visited.contains(buttons[i])) {
			continue;
		}
		visited.add(buttons[i]);
		String cur = "";
		for (int j = index; j <= i; j++) {
			cur += buttons[j];
		}
		helper(visited, result, buttons, sb.append(cur).append("-"), totalTime, i + 1, curTime + 1);
		visited.remove(buttons[i]);
	}
}
public List<String> combi (String input) {
		List<String> result = new ArrayList<>();
		boolean[] visited = new boolean[input.length()];
		helperCombi (visited, input, result, new StringBuilder());
		return result;
	}
	public void helperCombi (boolean[] visited, String input, List<String> result, StringBuilder sb) {
		if (sb.length() == input.length()) {
			result.add(new String(sb.toString()));
			return;
		}
		for (int i = 0; i < input.length(); i++) {
			if (visited[i]) {
				continue;
			}
			visited[i] = true;
			sb.append(input.charAt(i));
			helperCombi(visited, input, result, sb);
			sb.deleteCharAt(sb.length() - 1);
			visited[i] = false;
		}
	}








public int maxProduct(int[] nums) {
	int[] min = new int[nums.length];
	int[] max = new int[nums.length];
	min[0] = max[0] = nums[0];
	int result = nums[0];

	for (int i = 1; i < nums.length; i++) {
		if (nums[i] > 0) {
			min[i] = Math.min(nums[i] * nums[i], min[i - 1])
			max[i] = Math.max(nums[i] * nums[i], max[i - 1]);
		} else if (nums[i] < 0) {
			min[i] = Math.min(nums[i], max[i - 1] * nums[i]);;
			max[i] = Math.max(nums[i], min[i - 1] * nums[i]);;
		}
		result = Math.max(max[i], result);
	}        
	return result;
}



1-2-3-4-5


class Replace {
	int pos;
	String origin;
	String now;
	Replace (int pos, String origin, String now) {
		this.pos = pos;
		this.origin = origin;
		this.now = now;
	}
}
public String replacement (String file, Replace[] replaces) {
	int start = 0;
	StringBuilder sb = new StringBuilder();

	for (Replace cur : replaces) {
		int pos = cur.pos;
		String o = cur.origin;
		String now = cur.now;

		if (start < pos) {
			sb.append(file.substring(start, pos));
		}

		sb.append(now);
		start += o.length();
	}
	if (start < file.length()) {
		sb.append(start, sb.length());
	}
	return sb.toString();
}

class Query {
	public boolean TermQuery(String word) {

	}
	public boolean PhraseQuery (String phrase) {

	}
	public boolean OR (String str1, Stirng str2) {

	}
	public boolean AND () {

	}
}




public List<Integer> findSubstring(String s, String[] words) {
	List<Integer> result = new ArrayList<>();
	//store the word frequency in map
	Map<String, Integer> map = new HashMap<>();
	
	int totalLen = 0;
	for (String word : words) {
		map.put(word, map.containsKey(word) ? map.get(word) + 1 : 1);
		totalLen += word.length();
	}
	int wordLen = words[0].length();
	int start = 0, count = 0;

	for (int i = 0; i < wordLen; i++) {
		Map<String, Integer> temp = new HashMap<>();
		int start = i, count = 0;
		for (int j = i; j <= s.length() - wordLen; j++) {
			String curWord = s.substring(j, j + wordLen);
			if (!map.containsKey(curWord)) {
				temp.clear();
				count = 0;
				start = j + len;
			} else {
				temp.put(curWord, temp.containsKey(curWord) ? temp.get(curWord) + 1 : 1);
				count++;
				while (temp.get(curWord) > map.get(curWord)) {
					String leftMost = s.susbtring(start, start + len);
					temp.put(leftMost, temp.get(leftMost) - 1);
					start = start + len;
					count--;
				}
				if (count == words.length) {
					String leftMost = s.substring(start, start + len);
					temp.put(leftMost, temp.get(leftMost) - 1);
					result.add(start);
					start = start + len;
					count--;
				}
			}
		}
	}
	return result;


}


public int longestConsecutive (int[] nums) {
	if (nums == null || nums.length == 0) {
		return 0;
	}
	if (nums.length == 1) {
		return nums.length;
	}
	int maxLen = 0;
	int start = 0;
	int preVal = nums[start];
	for (int i = 1; i < nums.length; i++) {
		while (i < nums.length && nums[i] == preVal + 1) {
			preVal = nums[i];
			i++;
		}
		if (i == nums.length) {
			return nums.length - start;
		}
		maxLen = Math.max(maxLen, i - start);
		start = i;
	}
	return maxLen;
}


int maxLen = 0;
public int longestConsecutive (TreeNode root) {
	helper(root, 0);
	return maxLen;
}

public int helper (TreeNode root, int depth) {
	if (root == null) {
		return depth;
	}
	if (root.left == null && root.right == null) {
		return 1;
	}
	int left = helper(root.left, depth);
	int right = helper(root.right, depth);
	maxLen = Math.max(left, right);

	if (root.left != null) {
		if (root.left.val == root.val + 1) {
			left++;
		} else {
			left = 0;
		}
	}
	if (root.right != null) {
		if (root.right.val == root.val + 1) {
			right++;
		} else {
			right = 0;
		}
	}
	if (left == right && left == 0) {
		return 1;
	} else {
		return Math.max(left, right);
	}
}





class Node {
	boolean isMatch;
	int num;
	Node (boolean isMatch, int num) {
		this.isMatch = isMatch;
		this.num = num;
	}
}
public Node (String word, String guess) {
	
	if (word.equals(guess)) {
		return new Node (true, words.size());
	} else {
		int count = 0;
		for (int i = 0; i < Math.min(guess.length(), word.length(); i++) {
			if (guess.charAt(i) == word.charAt(i)) {
				possible.add(guess.charAt(i));
				count++;
			}
		}
		if (count == 0) {
			for (int i = 0; i < guess.length(); i++) {
				nPossible.add(guess.charAt(i));
			}
		}
		return new Node(false, count);
	}
}

Set<Character> possible = new HashSet<>();
Set<Character> nPossible = new HashSet<>();
Set<String> visited = new HashSet<>();
int[] scores = new int[words.size()];



public String getNext () {

}

public void scoreWord (List<String> words) {
	for (int i = 0; i < words.size(); i++) {
		if (visited.contains(words.get(i))) {
			continue;
		} else {
			int score = 0;
			for (char c : words.get(i).toCharArray()) {
				if (possible.contains(c)) {
					score++;
				} else if (nPossible.contains(c)) {
					score--;
				}
			}
			scores[i] = score;
		}
	}
}


int minStep = Integer.MAX_VALUE;
public int (int a, int b) {
	helper(a, b, 0, 0);
	return minStep;
}
// index [+1, -1, *2]
public int helper (int input, int target, int step, int index) {
	if (input < 0 || input > target) {
		return;
	}
	if (input == target) {
		minStep = Math.min(minStep, step);
		return;
	}
	for (int i = index; i < 3; i++) {
		if (i == 0) {
			helper(input + 1, target, step + 1, index);
		} else if () {
			helper(input - 1, target, step + 1, index);
		} else {
			helper(input * 2, target, step + 1, index);
		}
	}
}


public List<Integer> minSquare (int target) {
	List<Integer> result = new ArrayList<>();

	if (target <= 0) {
		return result;
	}
	while (target >= 0) {
		int left = 1, right = target;
		while (left + 1 < right) {
			int mid = left + (right - left) / 2;
			if (mid = target / mid) {
				result.add(mid);
				return result;
			}  else if (mid > target / mid) {
				right = mid;
			} else {
				left = mid;
			}
		}
		if (left < target / left) {
			target -= left * left;
		} else if (right < target / right) {
			target -= right * right;
		}
	}
	return result;
}

public List<Integer> minSquare (int target) {
	List<Integer> result = new ArrayList<>();

	int[] arrs = new int[target + 1];
	List<List<Integer>> result = new ArrayList<>();
	for (int i = 0; i <= target; i++) {
		result.add(new ArrayList<>());
	}
	int minLen = Integer.MAX_VALUE;
	int index = -1;
	for (int i = 1; i <= target; i++) {
		arrs[i] = Integer.MAX_VALUE;
		for (int i = 1; i <= target; i++) {
			if (arrs[target - i * i] != Integer.MAX_VALUE && target - i * i >= 0) {
				if (arrs[target - i * i] + 1 < arrs[i]) {
					arrs[i] = arrs[target - i * i];
					List<Integer> list = result.get(target - i * i);
					list.add(i);
					arrs[i] = list;
				}
				
			}
		}
		if (arrs[i] < minLen) {
			index = i;
		}
	}
	return result.get(index);
}


public List<Integer> findPermutaion (String str1, String str2) {
	int maxLen = str2.length();
	int[] chars = new int[26];
	List<Integer> result = new ArrayList<>();

	for (int i = 0; i < str2.length(); i++) {
		chars[str2.charAt(i) - 'a'] += 1; 
	}
	int totalChar = maxLen;
	int j = 0;
	for (int i = 0; i < str1.length - maxLen; i++) {

		while (j - i < maxLen - 1) {
			char c = str1.charAt(j);
			chars[c - 'a']--;
			if (chars[c - 'a'] >= 0) {
				totalChar--;
			}
			if (totalChar == 0) {
				result.add(i);
			}
			j++;
		}
		chars[str1.charAt(i) - 'a']++;
		if (chars[str1.charAt(i) - 'a'] > 0) {
			totalChar++;
		}
	}
	return result;
}



class Domino {
	private int left;
	private int right;

	Domino (int left, int right) {
		this.left = left;
		this.right = right;
	}

	public int getLeft () {
		return this.left;
	}
	public int getRight () {
		return this.right;
	}
}

class DominoBag {
	List<Domino> list;
	DominoBag (List<Domino> list) {
		this.list = list;
	}
	public void add (Domino d) {
		list.add(d);
	}
	public void remove(Domino d) {
		list.remove(d);
	}
}

public boolean findSequence (List<Domino> input, int start, int end) {
	

	for (int i = 0; i < input.size(); i++) {
		Domino cur = input.get(i);
		if (cur.getRight() == start) {
			//search from here to right
			if (existSubsequence(input, i, end, cur)) {
				return true;
			}
			
		} 
	}
	return false;
}
public boolean existSubsequence (List<Domino> input, int index, int target, Domino pre) {
	if (index >= input.size()) {
		return false;
	}
	boolean result = false;
	Domino cur = input.get(index);

	
		if (cur.getLeft() == target || cur.getRight() == target) {
			return true;
		}

		if (cur.getLeft() != cur.getRight()) {
			return false;
		} else {
			for (int i = index; i < input.size(); i++) {
				result |= existSubsequence(input, i, target, visited, cur);
			}
		}
	
	return result;

	}
}





int index = 0;
boolean result = 0;
public boolean existLoop (List<Domino> input) {
	while (index < input.size()) {
		Set<Integer> visited = new HashSet<>();
		if (helper(input, visited, null)) {
			return true;
		}
		index++;
	}
	return false;
}

public void helper (List<Domino> input, Set<Integer> visited, Domino pre) {
	if (index >= input.size()) {
		return;
	}
	Domino cur = input.get(index);
	if (visited.contains(cur.getRight()) {
		result = true;
		return;
	}
	if (pre != null) {
		if (pre.getRight() != cur.getLeft()) {
			return;
		}
	} 
	visited.add(cur.getLeft());
	visited.add(cur.getRight());
	index++;
	helper(input, visited, cur);
}



enum response {
	HIT, MISS, SUNK;
}
class Ship {
	Set<Integer> parts;
	void remove(int pos) {
		parts.remove(new Integer(pos));
	}
	void add (int pos) {
		parts.add(pos);
	}
	boolean hasSunk () {
		return parts == null || parts.size() == 0;
	}
}
class Player () {
	int id;
	Board gameBoard;
	Player(int id) {
		this.id;
	}
	Hit (int x, int y){
		if (!gameBoard.validShot(int pos)) {
			return response.MISS;
		} else {
			Ship ship = board.getShip ();
		}
	}
}
class Board {
	Map<Integer, Ship> map;
	Player player;
	boolean validShot (int pos) {
		return 
	}
	remove () {

	}
	Ship getShip () {

	}
}
public int countBattleships(char[][] board) {
	// 0 left 1 right 2 up 3 down
	for (int i = 0; i < board.length; i++) {
		for (int j = 0; j < board[0].length; j++) {
			if (board[i][j] == 'X') {
				helper(i, j, board, );
			}
		}
	}

}



final String target = "";
String guessWord (String word) {
	HashSet<Character> set = new HashSet<>();
	for (int i = 0; i < word.length(); i++) {
		set.add(word.charAt(i));
	}
	char[] result = new char[target.length()];
	for (int i = 0; i < target.length(); i++) {
		if (set.contains(target.charAt(i))) {
			result[i] = target.charAt(i);
		} 
	}
	return new String(result);
}
Set<Character> correct;
Set<Character> uncorrect;

//sort with high score; with correct and unvisited

(List<String> words, 
//first guessor with vowls 
Collections.sort(str, new Comparator<String>() {
	public int compare (String str1, String str2) {
		return score(str2) - score(str1);
	}
});
public int score (String str) {
	int score = 0;
	for (int i = 0; i < str.length(); i++) {
		swith (str.charAt(i)) {
			case 'a', case 'e', case 'i', case 'o', case 'u' : 
				score++;
		}
	}
	return score;
}
//sort words

class Formula {
	boolean val;
	Formula (String var) {
		if (var.equals("true")) {
			this.val = true;
		} else {
			this.val = false;
		}
	}
}

public Formula or(Formula f1, Formula f2) {
	if (f1.val || f2.val) {
		return new Formula("true");
	}
	return new Formula("false");
}

public Formula and(Formula f1, Formula f2) {
	if (f1.val == f2.val) {
		return new Formula(String.valueOf(f1.val));
	}
	return new Formula("false");
}

public Formula not(Formula f) {
	if (f.val) {
		return new Formula("false");
	}
	return new Formula("true");
}

public Formula init(String var) {
	return Formula(var);
}
// a || (b && c)
public boolean evaluate(Formula f, Map<String, Boolean> values) {

}


public int helper (TreeNode root, int depth, List<List<Integer>> result) {
	if (root == null) {
		return 0;
	}
	int left = helper(root.left, depth, result);
	int right = helper(root.right, depth, result)

	int level = Math.max(left, right);

	if (level >= result.size()) {
		List<Integer> list = new ArrayList<>();
		result.add(list);
	}
	result.get(level).add(root.val);

	return level + 1;
}




public int secondLargest (TreeNode root) {
	Integer first = null, second = null;
	helper(root, first, second);
	return second == null ? -1 : second;
}
public void helper (TreeNode root, Integer first, Integer second) {
	if (root == null) {
		return;
	}
	helper(root.right, first, second);
	if (first == null) {
		first = root.val;
	} else {
		second = root.val;
	}
	helper(root.left, first, second);
}



class Transaction {
	String payer;
	List<String> payee;
	int amount;
	Transaction (String payer, List<String> payee, int amount) {
		this.payer = payer;
		this.payee = payee;
		this.amount = amount;
	}
}
class Balance {
	String name;
	int amount;
	Balance (String name, int amount) {
		this.name = name;
		this.amount = amount;
	}
}

public List<Balance> printBalance (List<Transaction> input) {
	// store <name, balanced amount>
	List<Balance> result = new ArrayList<>();

	Map<String, Integer> map = new HashMap<>();

	for (Transaction t : input) {
		String payer = t.payer;
		List<String> payee = t.payee;
		//
		int perPerson = t.amount / payee.size();
		// store all the people into map
		if (!map.containsKey(payer)) {
			map.put(payer, 0);
		} 
		for (String name : payee) {
			if (!map.containsKey(name)) {
				map.(name, 0);
			} 
			if (!name.equals(payer)) {
				map.put(name, map.get(name) + perPerson);
			} else {
				map.put(name, map.get(name) - perPerson * (payee.size() - 1));
			}
		}
	}
	for (Map.Entry<String, Integer> entry : map.entrySet()) {
		Balance b = new Balance(entry.getKey(), entry.getValue());
		result.add(b);
	}
	return result;
}



// O(2^n) time and space
public List<List<Integer>> subsets (List<Integer> input) {
	List<List<Integer>> result = new ArrayList<>();
	if (input == null || input.size() == 0) {
		return result;
	}
	List<Integer> list = new ArrayList<>();
	Collections.sort(input);
	helper(input, result, list, 0);
	return result;
}
public void helper (List<Integer> input, List<List<Integer>> result, List<Integer> list, int index) {
	result.add(new ArrayList<>(list));

	for (int i = index; i < input.size(); i++) {
		// if there is duplicates one, skip
		if (i != 0 && input.get(i) == input.get(i - 1)) {
			continue;
		}
		list.add(input.get(index));
		helper (input, result, list, i + 1);
		list.remove(list.size() - 1);
	}

}








class DLLNode{
	DLLNode pre;
	DLLNode next;
	int val;
	int index;
	DLLNode (int val, int index) {
		this.val = val;
		this.pre = null
		this.next = null;
		this.index = index;
	}
}

popMax if duplicates, pop nearest one from top
1,4,2,3,5,4,3  
popMax 
1,4,2,3,4,3
popMax
1,4,2,3,3
popMax
1,2,3,3

class maxStack {
	DDLNode head;
	int index;
	PriorityQueue<DDLNode> heap;
	maxStack () {
		this.index = 0;
		this.head = null;
		this.heap = new PriorityQueue<>(new Comparator<DLLNode> () {
			public int compare (DDLNode n1, DDLNode n2) {
				if (n1.val == n2.val) {
					// index decending
					return n2.index - n1.index;
				}
				// decending
				return n2.val - n1.val;
			}
		});
	}
	public void insert (int num) {
		if (head == null) {
			head = new DDLNode(num, index);
		} else {
			DDLNode node = new DDLNode(num, index);
			head.next = node;
			node.pre = head;
			head = node;
		}
		index++;
		heap.offer(head);
	}

	public int peekMax () {
		if (head == null) {
			return -1;
		}
	}
	public int popMax () {
		if (head == null) {
			return -1;
		}
		DDLNode max = heap.pop();
		index--;
		if (heap.isEmpty()) {
			head = null;
			return max.val;
		}
		max.pre.next = max.next;
		max.next.pre = max.pre;

	}
	public int pop() {
		if (head == null) {
			return -1;
		}
		int result = head.val;
		DDLNode node = head.pre;

		heap.remove(head);
		index--;

		if (node == null) {
			head = null;
			return result;
		}
		head = node;
		head.next = null;
		return result;
	}
}









import java.io.*;
import java.util.*;

public class Solution {
    public static Iterable<Integer> intersection(Iterator<Integer> a, Iterator<Integer> b) {
        List<Integer> result = new ArrayList<>();
        . visit 1point3acres.com for more.
        if (!a.hasNext() || !b.hasNext()) {
            return result;
        }
        
        Integer currA = a.next();
        Integer currB = b.next();
        -google 1point3acres
        while (currA != null && currB != null) {
            if (currA.equals(currB)) {
                result.add(currA);
                
                if (a.hasNext()) {. 1point3acres.com/bbs
                    currA = a.next();
                } else {
-google 1point3acres                    currA = null;
                }
                
                if (b.hasNext()) {
                    currB = b.next();
                } else {
                    currB = null;
                }
            } else if (currA < currB) {
                if (a.hasNext()) {
                    currA = a.next();
                } else {
                    currA = null;
                }
            } else {
                if (b.hasNext()) {
                    currB = b.next();.鐣欏璁哄潧-涓€浜�-涓夊垎鍦�
                } else {
                    currB = null;
                }
            }
        }. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
        
        return result;
    }
    
    public static void main(String[] args) {
        List<Integer> a = new ArrayList<>();
        a.add(1);
        a.add(3);
        a.add(5);
        
        List<Integer> b = new ArrayList<>();
        b.add(1);
        b.add(2);
        b.add(3);
        b.add(5);
        b.add(6);
        
        Iterable<Integer> result = intersection(a.iterator(), b.iterator());-google 1point3acres
         鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�. 
        for (Integer num : result) {
            System.out.println(num);
        }
    }
}







class DLLNode{
	DLLNode pre;
	DLLNode next;
	int val;
	DLLNode (int val) {
		this.val = val;
		this.pre = null
		this.next = null;
	}
}
public maxStack {
	DLLNode head;
	DLLNode middle;
	int size;
	maxStack () {
		this.head = null;
		this.middle = null;
		this.size == 0
	}
	public void insert (int target) {
		if (head == null) {
			head = new DLLNode(target);
			middle = head;
		} else {
			DLLNode node = new DLLNode(target);
			head.next = node;
			node.pre = head;
			head = node;
			// when size is odd, middle shift to next
			if (size % 2 != 0) {
				middle = middle.next;
			}
		}
	}
	
	public int pop () {
		if (head == null) {
			return -1;
		}
		DLLNode pre = head.pre;
		//only one node exists
		if (pre == null) {
			head = null;
			middle = null;
			return head.val;
		} 
		int result = head.val;
		// when size is odd, middle shift to left
		if (size % 2 != 0) {
			head = pre;
			head.next = null;
			middle = middle.pre;
		}
		size--;
		return result;
	}

	public int popMid () {
		if (middle == null) {
			return -1;
		}
		// if only one node exists
		int result = middle.val;
		if (middle.pre == null) {
			head = null;
			middle = null;
			return result;
		}
		
		middle.pre.next = middle.next;
		middle.next.pre = middle.pre;
		if (size % 2 != 0) {
			middle = middle.pre;
		} else {
			middle = middle.next;
		}
		size--;
		return result;
	}

}


public boolean validBST (TreeNode root) {
	if (root == null) {
		return true;
	}
	return helper (root, Integer.MAX_VALUE, Integer.MIN_VALUE);
}
public boolean helepr (TreeNode root, int maxVal, int minVal) {
	if (root == null) {
		return true;
	}
	if (root.val >= maxVal || root.val <= minVal) {
		return false;
	}
	return helper(root.left, root.val, minVal) && helper(root.right, maxVal, root.val);
}


public int[] mergeTwoArray (int[] num1, int[] num2) {
	int i1 = 0, i2 = 0;
	int[] nums = new int[num1.length + num2.length - 1];
	while (i1 < num1.length && i2 < num2.length) {
		
		if (num1[i1] < num2[i2]) {
			nums[i1 + i2] = num1[i1++];
		} else {
			nums[i1 + i1] = num2[i2++];
		}
		
	}

	while (i1 < num1.length) {
		nums[i1 + i2 - 1] = nums[i1++];
	}
	while (i2 < num2.length) {
		nums[i1 + i1 - 1] = nums[i2++];
	}
	return nums;
}




class hangMan {
	final int maxStep = 6;
	int step;
	String word;
	Map<Character, Integer> map = new HashMap<>();

	boolean result = false;

	hangMan (String word) {
		this.steps = 0;
		this.word = word;
		for (int i = 0; i < word.length(); i++) {
			char c = word.charAt(i);
			map.put(c, map.containsKey(c) ? map.get(c) + 1 : 1);
		}
	}
	public boolean guess (String guess) {
		step++;
		for (int i = 0; i < guess.length(); i++) {
			char c = guess.charAt(i);
			if (map.containsKey(c)) {
				if (map.get(c) == 1) {
					map.remove(c);
				} else {
					map.put(c, map.get(c) - 1);
				}
			}
		}
		if (step < maxStep && map.size() == 0) {
			return true;
		}
	}
}



public Node deleteRec(Node root, int key) {
	if (root == null) {
		return root;
	}
	TreeNode node = root;
	
		if (node.val < key) {
			root.right = deleteRec(root.right, key);
		} else if (node.val > key) {
			root.left = deleteRec(root.left, key);
		} else {
			// node with only one child or no child
	        if (root.left == null)
	            return root.right;
	        else if (root.right == null)
	            return root.left;

	        // node with two children: Get the inorder successor (smallest
	        // in the right subtree)
	        root.key = minValue(root.right);

	        // Delete the inorder successor
	        root.right = deleteRec(root.right, root.key);
		}
	
	return root;

}
public int findMin (TreeNode root) {
	int minVal = root.val;
	while (root.left != null) {
		minVal = root.left.val;
		root = root.left;
	}
	return minVal;
}

/*Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. 
(each operation is counted as 1 step.)

You have the following 3 operations permitted on a word:

a) Insert a character
b) Delete a character
c) Replace a character
*/


public List<List<Integer>> fractions (int target) {
	List<Integer> list = new ArrayList<>();
	List<List<Integer>> list = new ArrayList<>();

	helper(target, result, list);
	return result;
}
public void helper (int target, List<List<Integer>> result, List<Integer> list) {
	
}





int[] outPos = new int[2];
// right(0), down(1), left(2), up(3)
int[][] dir = new int[][]{{0,1},{0,1},{-1,0},{0,-1}};
public int[] ballGame (char[][] board, int row, int col, boolean down, boolean up, boolean left, boolean right) {
	helper(board, row, col, 2);
	return outPos;
}

public void helper (char[][] board, int row, int col, int d) {
	if (row < 0 || row >= board.length || col < 0 || col >= board[0].length) {
		outPos[0] = row;
		outPos[1] = col;
		return;
	}
	
	if (board[row][col] == '') {
		//by default it goes down
		helper(board, row + 1, col, 2);
	} else {

	}
}


class twoSum {
	Map<Integer, Integer> map = new HashMap<>();
	public void add (int val) {
		if (map.containsKey(val)) {
			map.put(val, map.get(val) + 1);
		} else {
			map.put(val, 1);
		}
	}
	public boolean isExist (int val){
		for (int key : map.keySet()) {
			int left = val - key;
			if (key == left) {
				return map.get(key) > 1;
			} else {
				return map.containsKey(left);
			}
		}
		return false;
	}
}

class twoSum {
	Set<Integer> num = new HashSet<>();
	Set<Integer> sum = new HashSet<>();
	public void add (int val) {
		if (num.contains(val)) {
			return sum.add(val + val);
		} else {
			Iterator it = num.iterator();
			while (it.hasNext()) {
				sum.add(it.next() + val);
			}
		}
		num.add(val);
	}

	public boolean isExist (int val) {
		return sum.contains(val);
	}
}

public int maxDepth (TreeNode root) {
	if (root == null) {
		return maxDepth;
	}
	return helper(root, 1);
}
public int helper (TreeNode root, int depth) {
	if (root == null) {
		return depth - 1;
	}
	if (root.left == null && root.right == null) {
		return depth;
	}
	int result = Integer.MIN_VALUE;
	int left = helper(root.left, depth + 1);
	int right = helper(root.right, depth + 1);
	result = Math.max(left, right);
	return result;
}






import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class BoundedBlockingQueue<E> {

 private int capacity;
 private Queue<E> queue;
 public Lock lock = new reentrantLock();
 private Lock pushLock = new ReentrantLock();
 private Condition notFull = this.lock.newCondition();
 private Condition notEmpty = this.lock.newCondition();
    
 // only initialize this queue once and throws Exception if the user is
 // trying to initialize it multiple t times.
 public void init(int capacity) throws Exception {
     this.lock.lock();
     try{
         if(this.queue == null){
             this.queue = new LinkedList<>();
             this.capacity = capacity;
         } else {
             throw new Exception();
         }
     }finally{
         this.lock.unlock();
     }
 }

 // throws Exception if the queue is not initialized
 public void push(E obj) throws Exception {
     this.pushLock.lock();
      this.lock.lock();
     try{
         while(this.capacity == this.queue.size())
             this.notFull.wait();
         this.queue.add(obj);
         this.notEmpty.notifyAll();
     }finally{
         this.lock.unlock();
         this.pushLock.lock();
     }
 }

 // throws Exception if the queue is not initialized
 public E pop() throws Exception {
     this.lock.lock();
     try{
         while(this.capacity==0)
             this.notEmpty.wait();
         E result = this.queue.poll();
         notFull.notifyAll();
         return result;
     }finally{
         this.lock.unlock();
     }
 }

 // implement a atomic putList function which can put a list of object
 // atomically. By atomically i mean the objs in the list should next to each
 // other in the queue. The size of the list could be larger than the queue
 // capacity.
 // throws Exception if the queue is not initialized
 public void pushList(List<E> objs) throws Exception {
     this.pushLock.lock();
     this.lock.lock();
     try{
         for(E obj : objs){
             while(this.queue.size() == this.capacity)
                 this.notFull.wait();
             this.queue.add(obj);
             this.notEmpty.notifyAll();
         }
     }finally{
         this.lock.unlock();
         this.pushLock.unlock();
     }
 }
}







1,4,-8,4,-8,7
1,4,1,4,xxx,
1,4,-32,-128,


c1, c2
if (map1.containsKey(c1)) {
	if (map2.get(c2) != c1)  {
		return false;
	}
} else {
	if (map2.containsKey(c2)) {
		return false;
	} else {
		map1
		map2
	}
}

public int mySqrt(int x) {
	if (x < 0) {
		return -1;
	}
	int left = 1, right = x / 2;
	while (left + 1 < right) {
		int mid = left + (right - left) / 2;
		if (mid * mid < x) {
			left = mid;
		} else if (mid * mid > x) {
			right = mid;
		} else {
			return mid;
		}
	}


}

class Elem{
    public int value;
    public Elem max;
    public Elem next;
    public Elem pre;
 
    public Elem(int value, int min){
        this.value = value;
        this.max = min;
        this.next = null;
        this.pre = null;
    }
}
 
public class MinStack {
    public Elem head;
 	public Elem node;
    /** initialize your data structure here. */
    public MinStack() {
 		this.head = null;
    	this.node = head;
    }
 
    public void push(int x) {
        if (node == null){
            node = new Elem(x, x);
            node.max = head;
        }else{
            Elem e = new Elem(x, Math.max(x, node.min));
            node.max = x < node.min ? e : node;
            e.next = node;
            node.pre = e;
            node = e;
        }
 
    }
 
    public int popMax() {
        if (head.next == null) {
        	return -1;
        }
 		Elem max = head.next.max;
 		max.next.pre = max.pre;
 		max.pre.next = max.next;
 		return max.val;
    }
 
    public int peekMax() {
        if (head.next == null) {
        	return -1;
        }
        return head.next.max.val;
    }
 
    
}


public TreeNode LCA (TreeNode root, TreeNode t1, TreeNode t2) {
	if (root == null) {
		return null;
	}
	if (root == t1 || root == t2) {
		return root;
	}
	TreeNode left = LCA(root.left, t1, t2);
	TreeNode right = LCA(root.right, t1, t2);
	if (left != null && right != null) {
		return root;
	} else if (left != null) {
		return left;
	} else if (right != null) {
		return right;
	}
	return null;
}
public TreeNode LCA (TreeNode root, TreeNode t1, TreeNode t2) {
	int h1 = getHeight(root, t1);
	int h2 = getHeight(root, t2);
	while (h1 != h2) {
		if (h1 > h2) {
			t1 = t1.parent;
			h1 = getHeight(t1);
		} else if (h1 < h2) {
			t2 = t2.parent;
			h2 = getHeight(t2);
		}
	}
	TreeNode parent = null;
	while (parent != null) {
		if (t1.parent == t2.parent) {
			parent = t1.parent;
		} else {
			t1 = t1.parent;
			t2 = t2.parent;
		}
	}
	return parent;
}
public int getHeight(TreeNode root, TreeNode target) {
	
	return helper(root, target, 0);
}
public int helper(TreeNode root, TreeNode target, int level) {
	if (root == null) {
		return 0;
	}
	if (root == target) {
		return level;
	}
	return Math.max(helper(root.left, target, level + 1),helper(root.right, target, level + 1));
}


public int getNodeHeight(Node root, Node x, int height){
		if(root==null) return 0;
		if(root==x) return height;
		
		//check if the node is present in the left sub tree
		int level = getNodeHeight(root.left,x,height+1);
		//System.out.println(level);
		if(level!=0) return level;
		
		//check if the node is present in the right sub tree
		return getNodeHeight(root.right,x,height+1);
	}




if (index == word.length()) {
	return true;
}

if (row < 0 || row >= rowNum || col < 0 || col >= colNum || board[row][col] != word.charAt(index)) {
	return false;
}

board[i][j] ^= 255;
for (int i = 0; i < 4; i++) {
	int x
	int y 
	helper(x, y, index + 1);
}



public void {
	result.add(new ArrayList<>(list));
	for (int index = 0; i < nums.length; i++) {
		list.add();
		helper(index + 1);
		list.remove();

	}
}


public List<String> addOperators(String num, int target) {
	List<String> result = new ArrayList<>();
	if (num.length() <= 1 || Integer.parseInt(num) == target) {
		result.add(num);
		return result;
	}
	String str = "+-*/";
	for (int i = 0; i < num.length(); i++) {
		List<String> left = addOperators
		List<String> right		
		if () {

		}
	}
}
minWindow (String s, String t) {
	int[] count = new int[256];
	for (char c : s.toCharArray()) {
		count[c]++;
	}
	int start = 0, minLen = Integer.MAX_VALUE;
	String result = "";
	int counter = 0;
	for (int i = 0; i < s.length(); i++) {
		char c = s.charAt(i);
		if (count[c] > 0) {
			counter++;
		}
		count[c]--;
		while (counter == t.length()) {
			if (minLen > (i - start + 1)) {
				minLen = i - start + 1;
				result = s.substring(start, i + 1);
			}
			count[s.charAt(start)]++;
			if (count[s.charAt(start)] > 0) {
				counter--;
			}
			start++;
		}
	}
	return result;
}


int maxLen = 0, minLen = 0;
public List<String> letterCombinations (String digits, HashSet<String> words) {

}
brute force : get every word, and check if it exists in words dictionary
optimize : build a trie to store words, [minLen, maxLen] and check if word in trie

TrieNode root = buildTrie(words);
for (String word : words) {
	TrieNode temp = root;
	helper(result, temp, word);
}

return result;

TrieNode {
	char c;
	char[] children;
	String word;
	TrieNode () {

	}	
}

public void helepr (List<String> result, TrieNode root, String word) {
	if (word.length() == 0) {
		result.add(root.word);
		return;
	}
	for (int i = 0; i < digits.length(); i++) {
		for (char c : map.get(i).toCharArray()) {

			if (root.children[c] == null) {
				continue;
			} else {
				TrieNode tempNode = root;
				root = root.children[c];
				helper(result, root, temp.substring(1));
				root = tempNode;
			}

		}
	}
}






List<Integer> result = new ArrayList<>();
result.add(0);
TreeNode root = new TreeNode(nums[nums.length - 1]);
for (int i = nums.length - 2; i >= 0; i--) {
	int count = insertNode (root, nums[i]);
	result.add(count);
}
return Collections.reverse();

public int insertNode (TreeNode root, int val) {
	int thisCount = 0;

	while (true) {
		if (val <= root.val) {
			root.count++;
			if (root.left == null) {
				roto.left = new TreeNode(val);
				break;
			} else {
				root = root.left;
			}
		} else {
			thisCount += root.count;
			if (root.right == null) {
				root.right = new TreeNode(val);
				break;
			} else {
				root = root.right;
			}
		}
	}
	return thisCount;
}



public int maxSumSubmatrix(int[][] matrix, int k) {
	int result = 0;

	for (int col = 0; col < colNum; col++) {
		int[] sums = new int[rowNum];
		for (int c = col; c < colNum; c++) {
			for (int i = 0; i < rowNum; i++) {
				sums[i] += matrix[i][c];
			}

			Set<Integer> set = new TreeSet<>();
			set.add(0);

			int curSum = 0;

			for (int sum : sums) {
				curSum += sum;
				Integer num = set.celing(curSum - k);
				if (num != null) {
					result = Math.max(result, curSum - num);
				}
				set.add(curSum);
			}
			
		}
	}
	return result;
}
[7,2,5,8] m = 2
l  r    mid 
8  22    15
8  15    11
12  15   13
12  13   12
13  13   13

++++
public boolean canWind (String s) {
	Map<String, Boolean> map = new HashMap<>();
	return helper(s.toCharArray(), map);
}

public boolean helper(char[] chars, Map<String, Boolean> map) {
	for (int i = 1; i < chars.length; i++) {
		if (chars[i] == '+' && chars[i - 1] == '+') {
			chars[i] = '-';
			chars[i - 1] = '-';
			boolean t;
			String key = String.valueOf(chars);
			if (!map.containsKey(key)) {
				t = helper(chars, map);
				map.put(key, t);
			} else {
				t = map.get(key);
			}

			chars[i] = '+';
			chars[i - 1] = '+';

			if (!t) {
				return true;
			}

		}
	}
	return false;
}

public int lengthLongestPath(String input) {

	Map<Integer, Integer> map = new HashMap<>();
	map.put(0, 0);
	String[] strs = input.split("\n");

	int maxSize = 0;
	int size = 0;
	for (String str : strs) {
		int index = str.lastIndexOf("\t");
		
		if (str.contains(".")) {
			maxSize = Math.max(maxSize, size + str.length());
		} else {
			map.put(index + 1, map.get(index + 1) + 1 + str.length());
		}

	}
	return maxSize;
}






int maxSize = 0;
public int maxSubTree (TreeNode root, int[] range) {
	if (root == null) {
		return 0;
	}
	helper(root, range, 0);
	return maxSize;
}
public int helper (TreeNode root, int[] range, int size) {
	if (root == null) {
		return 0;
	}
	
	int left = helper(root.left, range, size);
	if (root.left != null && left == 0) {
		return 0;
	}

	int right = helper(root.right, range, size);
	if (root.right != null && right == 0) {
		return 0;
	}

	if (root.val >= range[0] && root.val <= range[1]) {
		size = Math.maX(left + right + 1, size);
		return size;
	} else {
		return 0;
	}
	
}


    [2,8]


    	      13
    (4) 7            16
   (3)5   10     12     18
 (1)3 (1)6	  11  14     20


class Interval {
	int start;
	int end;
	Interval (int start, int end) {
		this.start = start;
		this.end = end;
	}
}
public int (list<Interval> input, Interval range) {
	if (input == null || input.size() == 0) {
		return 0;
	}
	Collections.sort(input, new Comparator<Interval> (Interval i1, Interval i2) {
		public int compare (Interval i1, Interval i2) {
			return i1.start - i2.start;
		}
	});

	int min = Integer.MAX_VALUE;
	int count = 0;
	Interval pre = null;
	for (int i = 0; i < input.size(); i++) {
		Interval cur = input.get(i);
		if (cur.end > range.start) {

		}
	}

}




public int gcd(int num1, int num2) {
	if (num2 == 0) {
		return num1;
	}
	return gcd(num2, num1 % num2);
}


public void (int[] arrs, int k) {
	int len = arrs.length;
	int gcd = gcd(len, k);
	for (int i = 0; i < ; i++) {
		int j = i + gcd;
		int temp = arr[i];
		while (j < len) {
			arrs[i] = arrs[j];
			i = j;
			j += gcd;
		}
		arrs[i] = temp;
	}
}

//insert binay tree
public TreeNode insertNode (TreeNode root, TreeNode node) {
	if (root == null) {
		return node;
	}

	if (root.val > node.val) {
		root.left = insertNode(root.left, node);
	} else {
		root.right = insertNode(root.right, node);
	}
	return root;
}


public int (String input, int min, int max, int width, int height) {
	int left = min;
	int right = max;
	while (left + 1 < right) {
		int mid = left + (right - left) / 2;
		if (canFitWord(mid)) {
			left = mid;
		} else {
			 right = mid;
		}
	}
	if (canFitWord(chars, width, height, right)) {
		return right;
	}
	if (canFitWord(chars, width, height, left)) {
		return left;
	}
	return 0;
}
public boolean canFitWord (char[] chars, int width, int height, int fontSize) {
	int lineHeight = getHeight(fontSize);
	int rowNum = height / lineHeight;

	int w = 0;
	int[] info = new int[n];
	for (int i = 0; i < chars.length; i++) {
		info[i][0] = getWidth(chars[i])
		w += info[i][0];
	}
	//beyond the limited screen area
	if (w > width * rowNum) {
		return false;
	}

	int leftHeight = height;
	int leftWidth = width;
	int start = 0;
	
	w = 0;
	h = 0;
	for (int i = 0; i < chars.length; i++) {
		if (chars[i] == ' ') {
			//go to next new line
			if (w > leftWidth) {
				if (w > width) {
					return false;
				}
				leftWidth = width - w;
				rowNum--;
				if (rowNum < 0) {
					return false;
				}
			} else {
				leftWidth -= w;
			}
		} else {
			w += info[i];
		}
	}
	return true;
}


public int getWidth (char c, int fontSize) {

}
public int getHeight (int fontSize) {

}





public int[][] (int n, int candy) {
	int[][] board = new int[n][n];
	int index = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			Set<Integer> visited = new HashSet<>();
			while (true) {
				board[i][j] = Random.nextInt(candy);
				if (!set.contains(board[i][j])) {
					set.add(board[i][j]);
					if (isValid(board, i, j)) {
						break;
					}
				}
			}
		}
	}
	return board;
}
public boolean isValid (int[][] board, int row, int col) {
	if (row - 2 >= 0) {
		if (board[row - 1][col] == board[row][col] && board[row - 2][col] == board[row][col]) {
			return false;
		}
	}
	if (col - 2 >= 0) {
		if (board[row][col - 1] = board[row][col] && board[row][col - 2] == board[row][col]) {
			return false;
		}
	}
	return true;
}


public int maxArea (List<int[]> positions) {
	if (positions == null || positions.size() < 4) {
		return 0;
	}
	int maxArea = 0;
	Set<String> set = new HashSet<>();
	Set<String> visited = new HashSet<>();
	for (int[] pos : positions) {
		set.add(new String(pos[0] + pos[1] + ""));
	}

	for (int i = 0; i < positions.length - 1; i++) {
		int x1 = positions.get(i)[0];
		int y1 = positions.get(i)[1];
		for (int j = i + 1; j < positions.length; j++) {
			int x2 = positions.get(j)[0];
			int y2 = positions.get(j)[1];
			//memorized visited positions
			if (visited.contains(x1 + y2) && visited.contains(x2 + y1)) {
				continue;
			} else {
				if (set.contains(x1 + y2) && set.contains(y2 + x1)) {
					visited.add(x1 + y1 + x2 + y2);
					visited.add(x1 + y2 + x2 + y1);
					visited.add(x2 + y2 + x1 + y1);
					visited.add(x2 + y1 + x1 + y2);
					maxArea = Math.max(maxArea, (int)Math.abs(x1, x2) * Math.abs(y1, y2));
				} else {
					continue;
				}
			}
		}
	}
	return maxArea;
}


public boolean canWin(String s) {
	Map<String, Boolean> map = new HashMap<>();
	return helper (s.toCharArray(), map);
}
public boolean helper (char[] chars, Map<String, Boolean> map) {

	for (int i = 0; i < chars.length - 1; i++) {
		if (chars[i] == '+' && chars[i + 1] == '+') {
			chars[i] = '-';
			chars[i + 1] = '-';
			boolean temp;
			if (!map.containsKey(new String(chars))) {
				temp = helper(chars, map);
				map.put(new String(chars), temp);
			} else {
				temp = map.get(new String(chars));
			}
			chars[i] = '+';
			chars[i] = '+';


			if (temp == false) {
				return true;
			}
		}
	}
	return false;
}



public boolean canWin(String s) {
        HashMap<String, Boolean> map = new HashMap();
        return canWin(s.toCharArray(), map);
    }
    
    private boolean canWin(char[] chars, HashMap<String, Boolean> map) {
        for (int i = 1; i < chars.length; i++)
            if (chars[i] == '+' && chars[i - 1] == '+') {
                chars[i] = '-'; 
                chars[i - 1] = '-';
                
                boolean t;
                String key = String.valueOf(chars);
                
                if (!map.containsKey(key)) {
                    t = canWin(chars, map);
                    map.put(key, t);      //System.out.println(key + " --> " + t);
                } else {
                    t = map.get(key);     //System.out.println(key + " ==> " + t);
                }   // can not directly use if (t) return true here, cuz you need to restore
                
                chars[i] = '+'; 
                chars[i - 1] = '+';
                
                if (!t) return true;
            }
        return false;
    }

public double[] calculation (String[][] equations, String[][] queries) {
	double[] result = new double[queries.length];
	Set<String> exist = new HashSet<>();
	for (Stirng[] eq : equations) {
		set.add(eq[0]);
		set.add(eq[1]);
	}
	for (int i = 0; i < queries.length; i++) {
		if (!set.contains(queries[i][0]) || !set.contains(queries[i][1])) {
			result[i] = -1.0d;
		} else {
			Set<Integer> visited = new HashSet<>();
			result[i] = helper(equations, queries[i], visited);
		}
	}

	return result;
}
public double helper (String[][] equations, String[] query, Set<Integer> visited) {
	
	for (String[] eq : equations) {
		if (eq[0].equals(query[0]) && eq[1].equals(query[1])) {
			return Double.parseDouble(eq[2]);
		} else if (eq[0].equals(query[1]) && eq[1].equals(query[0])) {
			return 1 / (Double.parseDouble(eq[2]))
		}
	}

	for (int i = 0; i < equations.length; i++) {
		if (!set.contains(i) && equations[i][0].equals(query[0])) {
			set.add(i);
			double temp = equations[i][2] * helper(equations, new String[]{equations[i][1], query[1]}, visited);
			if (temp < 0) {
				set.remove(new Integer(i));
			} else {
				return temp;
			}
			
		}
		if (!set.contains(i) && equations[i][1].equals(query[1])) {
			set.add(i);
			double temp = helper (equations, new String[]{equations[i][0], query[1]}, visited) / equations[i][2];
			if (temp < 0) {
				set.remove(new Integer(i));
			} else {
				return temp;
			}
		}
	}
	return -1.0d;
}

//number of island ii

public List<Integer> numIslands2(int m, int n, int[][] positions) {
	int[] arrs = new int[m * n];
	Arrays.fill(arrs, -1);
	int count = 0;
	List<Integer> result = new ArrayList<>();
	int[][] dir = new int[]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
	
	for (int[] pos : positions) {
		int x = pos[0];
		int y = pos[1];
		int index = x * n + y;
		arrs[index] = index;
		count++;

		for (int i = 0; i < 4; i++) {
			int dx = x + dir[i][0];
			int dy = y + dir[i][1];
			if (dx >= 0 && dx < m && dy >= 0 && dy < n && arrs[dx * n + dy] != -1) {
				int root = dx * n + dy;
				int pos = find(arrs, dx * n + dy);
				if (root != arrs[root]) {
					arrs[root] = pos;
					count--;
				}
			} 	
		}
		result.add(count);
	}
	return result;
}

public int find (int[] arrs, int num) {
	if (num[num] == num) {
		return num;
	}
	int pos = find (arrs, arrs[num]);
	arrs[num] = pos;
	return pos;
}



public boolean validBST (TreeNode root) {
	if (root == null) {
		return true;
	}
	return helper (root, Integer.MIN_VALUE, Integer.MAX_VALUE;
}
public boolean helper (TreeNode root, int min, int max) {
	if (root == null) {
		return true;
	}
	if (root.val <= min || root.val >= max) {
		return false;
	}
	return helper(root.left, min, root.val) && helper(root.right, root.val, max);
}


public TreeNode find(TreeNode root, TreeNode t1, TreeNode t2) {
	//get the first list
	if (root == t1 || root == t2 || root == null) {
		return root;
	}

	TreeNode left = find (root.left, t1, t2);
	TreeNode right = find(root.right, t1, t2);

	if (left == null && right == null) {
		return null;
	} else if (left != null && right != null) {
		return root;
	} else if (left != null) {
		return left;
	} else {
		return right;
	}

}

int maxDepth = 1;
public TreeNode findCommonNode(TreeNode root) {
	//if max depth nodes == 1, than it is their parent node;
	// else find the lowest common ancestor
	List<List<TreeNode>> lists = new ArrayList<>();
	List<Integer> list = new ArrayList<>();
	helper (root, 1, list, lists);

	if (lists.size() == 1) {
		List<TreeNode> l = lists.get(0);
		return l.get(l.size() - 2);
	} 
	TreeNode result = null;

	int i = 0;
	while (i < Math.min(lists.get(0).size(), lists.get(1).size())) {
		if (lists.get(0).get(i) == lists.get(1).get(1)) {
			result = lists.get(0).get(i);
		} else {
			break;
		}
	}
	return result;
}
public void helepr (TreeNode root, int depth, List<TreeNode> list, List<List<TreeNode>> lists) {
	if (root.left == null && root.right == null) {
		if (depth == maxDepth) {
			lists.add(new ArrayList<>(list));
		} else if (depth > maxDepth) {
			maxDepth = depth;
			lists.clear();
			list.add(root);
			lists.add(new ArrayList<>(list));
			list.remove(list.size() - 1);
		}
		return;
	}
	for (TreeNode child : root.children) {
		list.add(child);
		helper (child, depth + 1, list, lists);
		list.remove(list.size() - 1);
	}
}


public class diceWords {
        interface find {
			boolean isAnyDice(int index);
        }
        public static boolean findWord(String[] words, String word){
                char[] arr = word.toCharArray(); boolean bb = true;
                int len = words.length;
                int[] tried = new int[len]; 
                int[] husband = new int[len];
                int[] wife = new int[word.length()];
                
                find f = new find(){
                        @Override
                        public boolean isAnyDice(int index) {
                                for (int j=0;j<len;j++){    //扫描每个骰子
                                           if (words[j].indexOf(arr[index])!=-1 && tried[j]==-1)      
                                        //如果骰子j有包含index的char并且还没有标记过，那么我就@@@“试图”@@@拿j做我老婆；
                                    //这个试图是跟这轮绑定的，每一轮以后清零；. Waral 鍗氬鏈夋洿澶氭枃绔�,
                                    //(这里标记的意思是这次查找曾试图改变过该骰子的归属问题，但是没有成功，所以就不用瞎费工夫了）. 
                                        {
                                                tried[j]=1;
                                                if (husband[j]==-1 || isAnyDice(husband[j])) { 
                                                        //骰子j没被抢了当老婆（没老公），那j是我的了！
                                                        //或者index的char能占其他骰子（isAnyDice(husband[j])实际上是isAnyOtherDice），那我还是可以把j当老婆，这里使用递归
                                                        husband[j]=index;
                                                        wife[index] = j;
                                                        return true;. visit 1point3acres.com for more.
                                                }
                                        }
                                }
                                return false;
                        }
                        
                };                        
                
                memset(husband,-1,len); 
                memset(wife,-1,word.length());     
                
                for (int i=0; i<arr.length; i++){
                        memset(tried,-1,len); //每一步清空tried；
                        bb = bb&&f.isAnyDice(i); 
                }
                for(int i=0; i<word.length(); i++){
                        System.out.println(wife[i]);//print couple!
                }
                return bb;
        } 
        private static void memset(int[] arr, int a, int len){
                for(int i=0; i<len; i++){
                        arr[i] = a;
                }
        } 
        public static void main(String[] args){
                String[] strarr = {"hewqed","doaefj","krelnv","fewqds"};
                boolean ha = false;. 
                ha = findWord(strarr,"khoq");
                System.out.print(ha);
        }
} 




public class Dijkstra {
	private static int M = 10000; //此路不通
	public static void main(String[] args) {
		//邻接矩阵 
		int[][] weight = {
		{0,10,M,30,100},
		{M,0,50,M,M},
		{M,M,0,M,10},
		{M,M,20,0,60},
		{M,M,M,M,0}
		};

		int start=0;
		int[] shortPath = dijkstra(weight,start);

		for(int i = 0;i < shortPath.length;i++)
		System.out.println("从"+start+"出发到"+i+"的最短距离为："+shortPath[i]);
	}

	public static int[] dijkstra(int[][] weight, int start) {
		//接受一个有向图的权重矩阵，和一个起点编号start（从0编号，顶点存在数组中） 
		//返回一个int[] 数组，表示从start到它的最短路径长度 
		int n = weight.length; //顶点个数
		int[] shortPath = new int[n]; //保存start到其他各点的最短路径
		String[] path = new String[n]; //保存start到其他各点最短路径的字符串表示
		for(int i = 0;i < n; i++)
			path[i]=new String(start + "-->" + 	i);
		int[] visited = new int[n]; //标记当前该顶点的最短路径是否已经求出,1表示已求出

		//初始化，第一个顶点已经求出
		shortPath[start] = 0;
		visited[start] = 1;

		for(int count = 1; count < n; count++) { //要加入n-1个顶点
			int k = -1; //选出一个距离初始顶点start最近的未标记顶点 
			int dmin = Integer.MAX_VALUE;
			for(int i = 0; i < n; i++) {
				if(visited[i] == 0 && weight[start][i] < dmin) {
					dmin = weight[start][i];
					k = i;
				}
			}

			//将新选出的顶点标记为已求出最短路径，且到start的最短路径就是dmin 
			shortPath[k] = dmin;
			visited[k] = 1;

			//以k为中间点，修正从start到未访问各点的距离 
			for(int i = 0; i < n; i++) {
				if(visited[i] == 0 && weight[start][k] + weight[k][i] < weight[start][i]) {
					weight[start][i] = weight[start][k] + weight[k][i];
					path[i] = path[k] + "-->" + i;
				}
			}
		}

		for(int i = 0; i < n; i++) {
			System.out.println("从"+start+"出发到"+i+"的最短路径为："+path[i]);
		}

		System.out.println("=====================================");
		return shortPath;
	}
}



public int numRnage (int[] arrs, int lower, int upper) {
	int[] sum = new int[arrs.length + 1];
	for (int i = 1; i <= arrs.length; i++) {
		sum[i] = sum[i - 1] + arrs[i - 1];
	}
	int posZero = helper(arrs, 0, arrs.length, 0);
	int result = 0;
	for (int i = 0; i < arrs.length; i++) {
		if (posZero != -1) {
			int left = helper(sum, i + 1, posZero, lower, true);
			int right = helper(sum, i + 1, posZero, upper, false);
			result += right - left;
		}
		int left = helper(sum, posZero, arrs.length, lower, true);
		int right = helper(sum, posZero, arrs.length, upper, false);
		result += right - left;
	}
	return result;
}
// binary search to find first val bigger or equal to target
public int helper (int[] arrs, int left, int right, int target, boolean findLower) {
	while (left + 1 < right) {
		int mid = left + (right - left) / 2;
		if (arrs[mid] < target) {
			left = mid;
		} else if (arrs[mid] > target){
			right = mid;
		} else {
			if (findLower) {
				right = mid;
			} else {
				left = mid;
			}
			
		}
	}
	if (findLower) {
		if (arrs[left] >= target) {
			return left;
		}
		if (arrs[right] >= 0) {
			return right;
		}
	} else {
		if (arrs[right] >= target) {
			return right;
		}
		if (arrs[left] >= target) {
			return left;
		}
	}
	
	return -1;
}

//create maze
class Path {
	int x;
	int y;
	Path (int x, int y) {
		this.x = x;
		this.y = y;
	}
}

int[][] dir = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
Map<Path, boolean> visited = new HashMap<>();
int n;
public List<Path> findAllPath (int n, Path start, Path end) {
	List<List<Path>> result = new ArrayList<>();
	this.n = n;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			Path cur = new Path(i, j);
			map.put(cur, false);
		}
	}	
	List<Path> path = new ArrayList<>();
	helper (start, end, path, result);
	return result;
}
public void helper (Path start, Path end, List<Path> path, List<List<Path>> result) {
	if (start.x == end.x && start.y == end.y) {
		result.add(new ArrayList<>(path));
		return;
	}
	if (start.x < 0 || start.x >= n || start.y < 0 || start.y >= n) {
		return;
	}
	if (visited.containsKey(start)) {
		return;
	}
	visited.put(start, true);
	path.add(start);
	
	for (int[] d : dir) {
		int x = d[0] + start.x;
		int y = d[1] + start.y;
		helper(new Path(x, y), end, path, result);
	}

	visited.put(start, false);
	path.remove(path.size() - 1);

}


// 1 means obstalce, 0 means empty
class Node {
	int x;
	int y;
	Node (int x, int y) {
		this.x = x;
		this.y = y;
	}
}

class Pair {
	Node pos;
	int dir;
	Pair (Node pos, int dir) {
		this.pos = pos;
		this.dir = dir;
	}
}

public int shortestPath (int[][] field, Node start, Node end) {
	if (field == null || field.length == 0 || field[0].length == 0 || start == end) {
		return 0;
	}
	int[][] dir = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
	Map<Node, Integer> map = new HashMap<>();
	map.put(start, 0);

	Queue<Pair> queue = new LinkedList<>();
	for (int d = 0; d < 4; d++) {
		queue.offer(start, d);
	}
	
	
	while (!queue.isEmpty()) {
		Pair cur = queue.poll();

		numTurns = map.get(cur.pos);

		//if the cell in the direciton is empty
		while (pos != end && helper(field, cur, dir)) {
			cur.pos.x = pos.x + dir[cur.dir % 4][0];
			cur.pos.y = pos.y + dir[cur.dir % 4][1]
			continue;
		}
		
		// find the end pos
		if (cur.pos == end) {
			return numTurns;
		}

		if (map.containsKey(cur.pos)) {
			continue;
		}

		queue.offer(new Pair(cur.pos, cur.dir + 1));
		queue.offer(new Pair(cur.pos, cur.dir + 3));

		map.put(cur.pos, numTurns + 1);
	}

	return -1;

}
public boolean helper (int[][] filed, Pair cur, int[][] dir) {
	int x = cur.x + dir[cur.dir % 4][0];
	int y = cur.y + dir[cur.dir % 4][1];

	if (x < 0 || x >= filed.length 
		|| y < 0 || y >= field[0].length || field[x][y] == 1) {
		return false;
	}
	
	return true;
}




Map<TreeNode, Integer> map = new HashMap<>();

public List<Integer> printSum (TreeNode[] nodes) {
	List<Integer> result = new ArrayList<>();
	if (nodes == null || nodes.length == 0) {
		return result;
	}
	helper(result, nodes, nodes[0], node[0].val);

	return map;
}
public void helper (TreeNode[] nodes, TreeNode node, int val) {
	if (node == root) {
		return;
	}
	if (map.containsKey(node)) {
		continue;
	}
	for (TreeNode parent : node.parent) {
		if (!map.containsKey(parent)) {
			map.put(parent, parent.val);
		} else {
			int curVal = map.get(parent);
			map.put(parent, curVal + parent.val);
		}
		helper(nodes, parent, parent.val);
	}
} 

class TrieNode {
	TrieNode[] children;
	int val;
	boolean isWord;
	TrieNode (int val) {
		this.val = val;
		this.children = new TrieNode[26];
		this.isWord = false;
	}
}
int[][] arrs;
public int longestSubsequence (String target, List<String> dict) {
	int len = target.length();
	if (len == 0) {
		return 0;
	}
	//build Trie Tree
	TrieNode root = buildTrie(dict, len);
	//generate pos map index : time: o(len of target)
	this.arrs = new int[target.length()][26];
	int maxLen = 0;

	for (int i = target.length() - 1; i >= 0; i--) {
		for (int j = 0; j < 26; j++) {
			if (target.charAt(i) - 'a' == j) {
				arrs[i][j] = i;	
			} else {
				if (i + 1 < target.length()) {
					arrs[i][j] == arrs[i + 1][j];
				} else {
					arrs[i][j] = -1;
				}
			}
		}
	}

	int maxLen = 0;
	helper (root, 0, 0, maxLen);
	return maxLen;

}
public void helper (TrieNode root, int start, int len, int maxLen) {
	if (root == null || start == -1) {
		return;
	}
	if (root.isWord) {
		maxLen = Math.max(maxLen, len);
	}
	for (char c = 'a'; c <= 'z'; c++) {
		helper (root.children[c - 'a'], arrs[start][c], len + 1, maxLen);
	}
}
public TreeNode buildTrie (List<String> dict, int maxDepth) {
	TreeNode result = new Node(-1);
	TreeNode node = result;
	for (String word : dict) {
		if (word.length() > maxDepth) {
			continue;
		}
		for (char c : word.toCharArray()) {
			if (node.children[c - 'a'] == null) {
				node.children[c - 'a'] = new TrieNode(c);
			}
			node = node.children[c - 'a'];
		}
		node.isWord = true;
	}
	return result;
}


public int maxSubsequence (List<String> dict, String target) {
  
	//预处理target，更新每一个char出现的pos
	int[][] arrs = new int[target.length()][26];
	int maxLen = 0;

	for (int i = target.length() - 1; i >= 0; i--) {
		for (int j = 0; j < 26; j++) {
			if (target.charAt(i) - 'a' == j) {
				arrs[i][j] = i;	
			} else {
				if (i + 1 < target.length()) {
					arrs[i][j] == arrs[i + 1][j];
				} else {
					arrs[i][j] = -1;
				}
			}
		}
	}

	for (String word : dict) {
		if (word.length() > target) {
			continue;
		}
		int pos = 0;
		for (char c : word.toCharArray()) {
			pos = arrs[pos][c];
			if (pos == -1) {
				break;
			} 
			pos = 
		}
		maxLen = Math.max(maxLen, word.length());
	}

	return maxLen;
}



public int[] createBIT (int[] input) {
	int[] BITree = new int[input.length + 1];
	for (int i = 0; i < input.length; i++) {
		int index = i + 1;
		while (index <= input.length) {
			BITree[index] += input[i];
			index += index & (-index);
		}
	}
	return BITree;
}
//want the sum 0 - index,
public int getSum (int[] BITree, int index) {
	int sum = 0;
	index++;
	while (index > 0) {
		sum += BITree[index];
		index -= index & (-index);
	}
	return sum;
}



public List<TreeNode> printNodes (TreeNode root, int target) {
	List<TreeNode> result = new ArrayList<>();
	if (root == null) {
		return result;
	}
	helper(result, root, target, true);
	return result;
}
public void helper (List<TreeNode> result, TreeNode root, int target, boolean appeared) {
	
	if (root == null) {
		return; 
	}
	
	if (root.val == target) {

		left = helper(result, root.left, target, true);
		right = helper(result, root.right, target, true);

	} else {
		if (appeared) 
			result.add(root.val);

		left = helper(result, root.left, target, false);
		right = helper(result, root.right, target, false);
	}
}





class DoubleLinkedNode {
	DoubleLinkedNode left;
	DoubleLinkedNode right;
	int val;
	DoubleLinkedNode (int val) {
		this.val = val;
		this.left = null;
		this.right = null;
	}
}
public int getNumOfNodes (DoubleLinkedNode root, List<DoubleLinkedNode> input) {
	Set<DoubleLinkedNode> set = new HashSet<>();
	for (root != null) {
		set.add(root);
		root.next;
	}
	Set<DoubleLinkedNode> inputSet = new HashSet<>();
	for (DoubleLinkedNode node : input) {
		inputSet.add(node);
	}

	int result = 0;
	while (inputSet.size() > 0) {
		for (DoubleLinkedNode node : input) {
			//left
			if (set.contains(node)) {
				result++;
				inputSet.remove(node);
				//left
				int temp = node;
				while (inputSet.contains(temp.left)) {
					inputSet.remove(temp.left);
					temp = temp.left;
				}
				temp = node;
				while (inputSet.contains(temp.right)) {
					inputSet.remove(temp.right);
					temp = temp.right;
				}
				
			}
			//right
		}
	}
	return result;
}





public List<List<Integer>> getPath (TreeNode root, int target) {
	List<List<Integer>> result = new ArrayList<>();
	if (root == null) {
		return result;
	}
	List<Integer> temp = new ArrayList<>();
	List<Integer> list = new ArrayList<>();
	helper(result, list, temp, root, target);
	return result;
}
public void helepr (List<List<Integer>> result, List<Integer> list, List<Integer> temp, TreeNode root, int target) {
	if (root == null) {
		return;
	}
	if (root.val == target) {
		for (int num : temp) {
			result.add(num);
		}
	}
	if (root.left == null && root.right == null) {
		list.add(root.val);
		result.add(new ArrayList<>(list));
		list.add(list.remove() - 1);
	}

	list.add(root.val);
	result.add(root.val);
	temp.add(root.val);
	
	helper(result, list, temp, root.left, target);
	helper(result, list, temp, root.right, target);
	
	list.remove(list.size() - 1);
	result.remove(list.size() - 1);
	temp.remove(temp.size() - 1);

}



public class ZigzagIterator {
    
    List<Iterator> list;
    int index;
    public ZigzagIterator(List<List<Integer>> input) {
        this.list = new ArrayList<>();
        for (List<Integer> in : input) {
        	if (in.size() > 0) {
        		list.add(in.iterator());
        	}
        }
        this.index = 0;
    }

    public int next() {
    	int result = -1;
    	if (!hasNext()) {
    		return result;
    	}
        
        int pos = index % list.size();

        result = list.get(pos).next();
        if (!list.get(pos).hasNext()) {
        	list.remove(pos);
        	index--;
        }
        index++;

        return result;
    }

    public boolean hasNext() {
        return list.size() > 0;
    }
}

/**
 * Your ZigzagIterator object will be instantiated and called as such:
 * ZigzagIterator i = new ZigzagIterator(v1, v2);
 * while (i.hasNext()) v[f()] = i.next();
 */











int first = a;
int second = b;
public List<Integer> getFibonacci() {
	List<Integer> result = new ArrayList<>();
	result.add(first);
	while (a > b) {
		result.add(b);
		int temp = first - second;
		a = b;
		b = temp;
	}
	return result;
}



public boolean isSubtree (TreeNode root, TreeNode target) {

	if (root == null) {
		return false;
	}
	if (target == null) {
		return true;
	}
 	
	if (isSame(root, target)) {
		return true;
	}
	return helper (root.left, target) || helper(root.right, target);
}
public boolean isSame (TreeNode root, TreeNode target) {
	if (root == null && target == null) {
		return true;
	}
	if (root == null || target == null) {
		return false;
	}
	if (root.val != target.val) {
		return false;
	}
	return isSame (root.left, target.left) && isSame (root.right, target.right);
}



public int rob(int[] nums) {
        // 求两种条件下更大的那个，用一个offset表示是哪种条件
        return Math.max(rob(nums, 0), rob(nums, 1));
    }
    
    public int rob(int[] nums, int offset) {
        // 如果长度过小，则直接返回结果
        if(nums.length <= 1 + offset){
            return nums.length <= offset ? 0 : nums[0 + offset]; 
        }
        int a = nums[0 + offset];
        // 如果offset是1，则从下标为1的元素开始计算，所以要比较nums[1]和nums[2]
        int b = Math.max(nums[0 + offset], nums[1 + offset]);
        // 对于不抢劫最后一个房子的情况，i要小于nums.length - 1
        for(int i = 2 + offset; i < nums.length - 1 + offset; i++){
            int tmp = b;
            b = Math.max(a + nums[i], b);
            a = tmp;
        }
        return b;
    }



//Kth Largest in BST
public TreeNode kthLargest(TreeNode root, int k) {
  if (root == null) {
  	return null;
  }
  TreeNode result = null;
  helper(root, k, 0, result);
  return result;
}
public void helper (TreeNode root, int k, int count, TreeNode result) {
	if (root == null || c > k) {
		return;
	}
	helper (root.right, k, count, result);
	count++;
	if (count == k) {
		result = root;
	
	}
	helper(root.left, k, count, result);
}


//merge sort
public ListNode mergeSort (ListNode root) {

	helper(root);
}
public 
fast 
slow
while () {

}
ListNode l1 = head;
ListNode l2 = slow.next;
slow.next = null;

public merge


public List<List<Integer>> findPath (TreeNode root, int target) {
	List<List<Integer>> result = new ArrayList<>();
	if (root == null) {
		return result;
	}
	List<Integer> list = new ArrayList<>(); 
	helper(result, list, root, target);
	return result;
}
public void helper (List<List<Integer>> result, List<Integer> list, TreeNode root, int target) {
	if (root == null || target < 0) {
		return;
	}
	if (target == 0) {
		result.add(new ArrayList<>(list));
		return;
	}
	list.add(root.val);
	helper(result, list, root.left, target - root.val);
	helper(result, list, root.right, target - root.val);
	list.remove(list.size() - 1);
}


public int NthPrime(int n){

	List<Integer> list = new ArrayList<>();
	list.add(2);
	int number = 3;
	while (list.size() < n) {
		boolean isPrime = true;
		for (int i = 0; i < list.size(); i++) {
			if (number % list.get(i) == 0) {
				isPrime = false;
			}


		if (isPrime) {
			list.add(number);
		}
		number += 2;
	}
	return list.get(list.size() - 1);
}

// find the mimum sum path from right bottom to up left corner which cloest to 0
public int numDecodings(int[][] matrix) {

	int rowNum = matrix.length;
	int colNum = matrix[0].length;

	int[][] filed = new int[rowNum][colNum];
	for (int j = 0; j < colNum; j++) {
		if (j == 0) {
			if (matrix[0][j] < 0) {
				filed[0][j] = -matrix[0][j];
			}
		} else {
			if (matrix[0][j] < 0) {
				matrix[0][j] = matrix[0][j - 1] + (-1 * matrix[0][j]); 
			} else {
				matrix[0][j] = matrix[0][j - 1];
			}
		}
	}

	for (int i =) {
		for () {
			if (matrix[i][j] < 0) {
				filed[i][j] = Math.min(filed[i - 1][j], fild[i][j - 1]) + (-1 * matrix[i][j]);
			} else {
				fild[][] = Math.min
			}
		}
	}

}





public int longestIncreasingContinuousSubsequence(int[] A) {

	int maxDecrease = Integer.MIN_VALUE;
	int maxIncrease = Integer.MIN_VALUE;
	//flag true: increase
	// falg false : decrease;
	int curIn = 1;
	int curDe = 1;
	boolean flag = false;

	if (A.length <= 1) {
		return A.length;
	}
	for (int i = 1; i < A.length; i++) {
		if (A[i] > A[i - 1]) {
			if (flag) {
				curIn += 1;
			} else {
				flag = true;
				maxDecrease = Math.max(curDe, maxDecrease);
				curDe = 2;
			}
		} else if (A[i] < A[i - 1]) {
			if (!flag) {
				curDe += 1;
			} else {
				flag = false;
				maxIncrease = Math.max(curIn, maxIncrease);
				curIn = 2;
			}
		}
	}
	if (curIn > maxIncrease) {

	}
	if (curDe > maxDecrease) {

	}
	return Math.max
}



class PeekingIterator implements Iterator<Integer> {

	Iterator<Integer> it;
	Integer temp = null;

	PeekingIterator (List<Integer> list) {
		this.it = list.iterator();
	}

	public int next () {
		if (temp != null) {
			int result = temp.intValue();
			temp = null;
			return result;
		}
		
		return it.next();
		
	}

	public boolean hasNext () {
		if (temp != null) {
			return true;
		}
		return it.hasNext();
	}
	public int peek() {
		if (temp == null && it.hasNext()) {
			temp = it.next();
		}
		return temp;
	}
}



public int left_size = 0;
public RankNode left;
public RankNode right;
public int data = 0;
public RankNode (int num) {
	this.data = num;
}
public void insert (int num) {
	if (data < num) {
		if (left != null) {
			left.insert(num);
		} else {
			left = new RankNode(num);
		}
		left_size++;
	} else {
		if (right != null) {
			right.insert(num);
		} else {
			right = new RankNode(num);
		}

	}
}



public ListNode removeElements(ListNode head, int val) {
	if (head == null) {
		return head;
	}
	ListNode dummy = new Listnode(0);
	dummy.next = head;
	ListNode node = dummy;

	while (node.next != null) {
		if (node.next.val == val) {
			node.next = node.next.next;
		}
		node = node.next;
	}
	return dummy.next;
}


public int maxProfit (int[] prices) {
	if (prices == null || prices.length <= 1) {
		return 0;
	}
	int pre = prices[0];
	int result = 0;
	for (int i = 1; i < prices.length; i++) {
		if (prices[i] > pre) {
			result += prices[i] - pre;
		} 
		pre = prices[i];
	}
	return result;
}



helper (int m, int n, int begin, int count, int[][] arrs, boolean[] visited) {
	if (count >= m) {
		result++;
	}
	if (count > n) {
		return;
	}
	for (int i = 1; i <= 9; i++) {
		if (viisted[i]) {
			continue;
		}
		int crossNum = arrs[begin][i];
		if (!visited[crossNum] && crossNum != 0) {
			continue;
		}
		visited[crossNum] = true;
		helper(m, n, i, count + 1, arrs, visited);
		visited[crossNum] = false;
	}
}


public class Bank {
	private int count = 0;

	public void checkBalance () {
		System.out.prinltn("account balance: " + count);
	}
	public void addMoney (int num) {
		count += num;
		System.out.println(System.currentTimeMillis() + "add money " + num);
	}
	public void getMoney (int num) {
		if (count - num < 0) {
			System.out.println("do not have enough money!");
		} 
		count -= num;
		System.out.println(System.currentTimeMillis + "account balance  " + count);
	}	
}



public List<String> wordBreak (String s, Set<String> wordDict) {
	List<String> [] pos = new List<Stirng>[s.length() + 1];
	pos[0] = new ArrayList<>();


	for (int i = 0; i < s.length(); i++) {
		if (pos[i] != null) {
			for (int j = i + 1; j <= maxLen; j++) {
				String sub = s.substring(i, i + j);
				if (wordDict.contains(sub)) {
					if (pos[j] != null) {
						List<String> temp = pos[j];
					} else {
						List<Strint> temp = new ArrayList<>();
					}
					temp.add(sub);
				}
			}
		}
	}

	List<String> result = new ArayList<>();
	if (pos[s.length()] == null) {
		return result;
	} 
	helper (result, "", s.length(), pos);
	return result;	
	
}
public void helper (List<String> result, String str, int index, List<String>[] pos) {
	if (index < 0) {
		return;
	}
	if (pos[index] == null) {
		return;
	}
	if (index == 0) {
		result.add(str);
		return;
	}

	for (String s : pos[index]) {
		helper(result, s + str, index - s.length(), pos);
	}
	
}

//binary tree里面最长的path node的数量
int max;
public int lenOfNodes (TreeNode root) {
	if (root == null) {
		return 0;
	}
	max = 1;
	helper(root.val);
	return max;
}
public int helper(TreeNode root) {
	if (root == null) {
		return 0;
	}
	int left = helper(root.left);
	int right = helper(root.right);

	int temp = Math.max(left, right) + 1;
	max = Math.max(max, Math.max(temp, left + right + 1));
	return temp;
}

public class LinkedList<T> {
	private Node head;
	private int size;
	private Object lock;
	public LinkedList() {
		this.size = 0;
		this.head = null;
	}
	public int size() {
		return this.size;
	}
	public void add (T object) {
		synchronized(lock) {
			Node node = new Node(object);
			node.next = this.head;
			this.head = node;
			this.size++;
		}		
	}
	public T get(int index) throws Exception {
		Node head = this.head;
		for (int i = 0; i < this.size; i++) {
			if (i == index) {
				return (T) head.data;
			} else {
				head = head.next;
			}
		}
		throw new Exception("Item not found.");
	}
}
class Node {
	public Object data;
	public Node next;
	Node (Object data) {
		this.next = null;
		this.data = data;
	}
}


public List<Integer> leafNode (TreeNode root) {
	List<Integer> list = new ArrayList<>();
	helper(list, root);
	return list;
}
public void helper (List<Integer> list, TreeNode root) {
	if (root == null) {
		return;
	}
	if (root.left == null || root.right == null) {
		list.add(root.val);
		return;
	}
	helper(list, root.left);
	helper(list, root.right);
}

boolean isExist = false;
public boolean isExist (int[][] matrix, int target) {
	
	int row = 0;
	int col = matrix[0].length - 1;

	while (row < matrix.length && j >= 0) {
		int cur = matrix[row][col];
		if (cur == target) {
			return true;
		} else if (cur > target) {
			col--;
		} else {
			row++;
		}
	}
	return false;
}
// This, is a - test
// test, a is - This



public LinkedList reverse (LinkedList head) {
	LinkedList node = null;
	while (head != null) {
		if (node != null) {
			LinkedList temp = new LinkedList(head.val);
			temp.next = node;
			node = temp;
		} else {
			node = new LinkedList(head.val);
		}
		head = head.next;
	}
	return node;
}

double[] nums = new double[4];
boolean flag;
double error = 1E-6;

public int can24 (int a, int b, int c, int d) {
	nums[0] = (double) (a);
	nums[1] = (double) (b);
	nums[2] = (double) (c);
	nums[3] = (double) (d);
	flag = false;
	game24(4);
	if (flag) {
		return 1;
	}
	return 0;
}
public void game24 (int n) {
	if (n == 1) {
		if (Math.abs((int)nums[0] - 24) <= 0) {
			flag = true;
			return;
		}
	}
	if (flag) {
		return;
	}

	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			double a;
			double b;
			a = nums[i];
			b = nums[j];
			nums[j] = nums[n - 1];
		}
	}

}


class Data {
	int size = 0;
	int min;
	int max;
	boolean isBST;
	Data () {
		size = 0;
		min = Integer.MAX_VALUE;
		max = Integer.MIN_VALUE;
		isBST = true;
	}
}
int max = Integer.MIN_VALUE;
public int largestBSTSubtree (TreeNode root) {
	if (root == null) {
		return 0;
	}
	Data data = helper(root);
	return max.size;
}
public Data helper (TreeNode root) {
	if (root == null) {
		return new Data();
	}

	Data left = helper(root.left);
	Data right = helper (root.right);
	
	Data cur = new Data();

	if (!left.isBST || !right.isBST || left.max >= root.val || right.min <= root.val) {
		cur.isBST = false;
		cur.size = Math.max(left.size, right.size);
		return cur;
	}
	cur.isBST = true;
	cur.size = left.size + right.size + 1;
	cur.min = root.left != null ? left.min : root.val;
	cur.max = root.right != null ? right.max : root.val;

	return cur;
}
//实现hashtable
public int hashCode (char[] key, int hash_size) {
	long hashCode = 0;
	long baseV = 1;
	for (int i = key.length - 1; i >= 0; i--) {
		hashCode += key[i] * baseV;
		baseV = baseV * 33;
	}

	return (int) (hashCode % hash_size);
}
public ListNode[] rehashing (ListNode[] hashTable) {
	if (hashTable == null || hashTable.length == 0) {
		return null;
	}
	ListNode[] result = new ListNode[hashTable.length * 2];
	for (int i = 0; i < hashTable.length; i++) {
		while (hashTable[i] != null) {
			int index = (hashTable[i].val % result.length + result.length) % result.length;
			ListNode temp = result[index];
			if (temp == null) {
				temp = new ListNode(hashTable[i].val);
			} else {
				while (temp.next != null) {
					temp = temp.next;
				}
				temp.next = new ListNode(hashTable[i].val);
			}
		}
		hashTable[i] = hashTable[i].next;
	}
}


// BST from preorder 
public int preStart = 0;
public TreeNode constructTree(int[] preorder, int min, int max) {
	if (preStart > preorder.length) {
		return null;
	}
	int cur = preorder[preStart++];
	TreeNode root = null;
	if (min < cur && cur < max) {
		root = new TreeNode(cur);
		if (preStart < preorder.length) {
			root.left = constructTree(preordr, min, cur);
			root.right = constructTree(preorder, cur, max);
		}
	}
	return root;
}
//BST from PostOrder
int postStart = postOrder.length - 1;
public TreeNode constructTreeUtil(int postOrder[], int min, int max) {
    // Base case
    if (postIndex < 0) {
        return null;
    }
    TreeNode root = null;
    // If current element of post[] is in range, then
    // only it is part of current subtree
    int cur = postOrder[postStart--];
    if (min < cur && cur < max) {
        // Allocate memory for root of this subtree and decrement
        // *postIndex
        root = new TreeNode(cur);
        if (postStart > 0) {
            // All nodes which are in range {key..max} will go in 
            // right subtree, and first such node will be root of right
            // subtree
            root.right = constructTreeUtil(postOrder, cur, max);

            // Contruct the subtree under root
            // All nodes which are in range {min .. key} will go in left
            // subtree, and first such node will be root of left subtree.
            root.left = constructTreeUtil(postOrder, min, cur);
        }
    }
    return root;
}


//I have a "faux coat"
//[I, have, a, faux coat]
public List<String> toWordList (String str) {
	List<String> result = new ArrayList<>();
	if (str == null || str.length == 0) {
		return result;
	}
	boolean isQutoe = false;
	StringBuilder sb = new StringBuilder();
	for (int i = 0; i < str.length() ;i++) {
		char c = str.charAt(i);
		if (isQuote) {
			if (c == '"') {
				result.add(sb.toString());
				isQuote = false;
				sb = new StringBuilder();
			} else {
				sb.append(c);
			}
		} else {
			if (c == '"') {
				isQuote = true;
				continue;
			} 
			if (c == " ") {
				if (sb.length() == 0) {
					continue;
				}
				result.add(sb.toString());
				sb = new StringBuilder();
			} else {
				sb.append(c);
			}
		}

	}
	if (sb.length() != 0) {
		result.add(sb.toString());
	}
	return result;
}

public class PeekIterator implements Iterator {

	private final Iterator iterator;
	private T nextitem;
	public PeekIterator (Iterator iterator) {
		this.iterator = iterator;
	}
	@Override
	public boolean hasNext() {

	}
	@Override
	public T next () {

	}
	public T peek () {

	}
	public void remove () {

	}
}


class SegmentTreeNode {
	SegmentTreeNode left;
	SegmentTreeNode right;
	int max;
	int start;
	int end;
	public SegmentTreeNode (int start, int end, int max) {
		this.start = start;
		this.end = end;
		this.max = max;
		this.left = null;
		this.right = null;
	}
}
public SegmentTreeNode build(int[] nums) {

	return buildTree (nums, 0, nums.length - 1);

}
public SegmentTreeNode buildTree(int[] nums, int start, int end) {
	if (start > end) {
		return null;
	}
	SegmentTreeNode root = new SegmentTreeNode(start, end, Integer.MIN_VALUE);
	if (start != end) {
		int mid = start + (end - start) / 2;
		root.left = buildTree(nums, start, mid);
		root.right = buildTree(nums, mid + 1, end);
		root.max = Math.max(root.left.max, root.right.max);
	} else {
		root.max = nums[end];
	}
	return root;
}

public String base10To62(int num) {
	StringBuilder sb = new StringBuilder();
	while (num != 0) {
		sb.insert(0, map.charAt(num % 62));
		num /= 62;
	}
	if (sb.length() < 6) {
		sb.insert(0, "0");
	}
	return sb.toString();
}
public int base62To10 (String str){
	int n = 0;
	for (int i = 0; i < str.length(); i++) {
		n = n * 62 + convert(str.charAt(i));
	}
	return n;
}
public int convert (char c ) {
	if ('0' <= c && c <= '9') {
		return c -'0';
	} else if ('a' <= c && c <= 'z') {
		return c - 'a' + 10;
	} else {

	}
	return -1;
}

public List<String> generateAbbreviations(String word) {

	int len = word.length();
	for (int i = 0; i < Math.pow(2, word); i++) {
		int temp = i;
		int count = 0;
		String out = "";
		for (int j = 0; j < word.length(); j++) {
			if ((temp & 1) == 1) {
				count++;
				if (j == word.length() - 1) {
					out += String.valueOf(count);
				}
			} else {
				if (count != 0) {
					out += String.valueOf(count);
				}
				out += word.charAt(j);
			}
			temp >>= 1;
		}
		result.add(out);
	}
	return result;
}


class TrieNode {
	TrieNode[] children;
	int val;
	String word;
	TrieNode (int val) {
		this.word = "";
		this.val = val;
		this.children = new TrieNode[26];
	}
}
String[] map = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
public List<String> letterCombinations (String digits, HashSet<String> words) {
	List<String> result = new ArrayList<>();
	if (digits == null || digits.length() == 0) {
		return result;
	}
	TrieNode root = new TrieNode();
	for (String word : words) {
		addWord(word, root);
	}
	buildTrie (root);

	Queue<String> q = new LinkedList<>;
	Queue<TrieNode> qT = new LinkedList<>();
	q.offer("");

	for (int i = 0; i < digits.length(); i++) {
		char c = digits.charAt(i);
		int size = q.size();
		for (int i = 0; i < size; i++) {
			String curStr = q.poll();
			TrieNode curTrie = qT.poll();
			for (int j = 0; j < map[c - '0'].length(); j++) {
				int index = map[c - '0'].charAt(j) - 'a';
				if (curTrie.children[index] != null) {
					qT.offer(curTrie.children[index]);
					q.offer(temp + map[c - '0'].charAt(j));
				}
			}
		}
	}
	while (q.size() > 0) {
		word.add(q.poll());
	}
	return word;
}
public void buildTrie (String word, TrieNode root) {
	for (char c : word.toCharArray()) {
		if (root.children[c - '0'] == null) {
			root.children[c - '0'] = new TrieNode();
		}
		root = root.children[c - '0'];
	}
	root.word = word;
}



// find all distinct palindromic substrings

public List<String> findAllPal (String str) {

	List<String> result = new ArrayList<>();
	if (str == null || str.length() == 0) {
		return result;
	}

	for (int i = 0; i < str.length(); i++) {
		helper(str, i, 0, result);
		helper(str, i, 1, result);
	}
	return result;
}
public void helper (String str, int index, int diff, List<String> result) {
	
	int left = index;
	int right = index + diff;
	while (left >= 0 && right <= str.length() - 1) {
		char l = str.charAt(left);
		char r = str.charAt(right);
		if (l == r) {
			String cur = str.substring(left, right + 1);
			if (!result.contains(cur)) {
				result.add(str.substring(left, right + 1));
			}			
			left--;
			right++;
		}
	}
	
}

public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
	result 
	if (num == null || num.length == 0) {
		return result;
	}
	Arrays.sort(num);
	helper(num, 0, new ArrayList<>(), result);
	return result;
}
public void helper () {
	if (list.size() == num.length) {

	}
	for (int i = 0; i < num.length; i++) {
		if (used[i] || !used[i] && i - 1 > 0 && num[i] == num[i - 1]) {
			continue;
		}
		used[i] = true;
		list.add();
		helper();
		list.remove();
		used[i] = false;
	}
}

//Unique Word Abbreviation
Set<String> uniqueDict;
Map<String, String> abbrDict;

public ValidWordAbbr(String[] dictionary) {
	
	for (String dict : dictionary) {
		String abb = toAbb(dict);

	}
}
public boolean isUnique(String word) {

}
public String toAbb (String str) {
	if (str.length() <= 2) {
		return str;
	}
	return str.charAt(0) + "" + (str.length() - 2) + "" + str.charAt(str.length() - 1);
}
// maximum minimum path
int max = Integer.MIN_VALUE;

public int MaxMinPath (int[][] matrix) {
	if (matrix == null || matrix.length == 0 || matrix[0].length) {
		return 0;
	}
	int min = Integer.MAX_VALUE;
	helper(matrix, min, 0, 0);
	return max;
}
public void helper (int[][] matrix, int min, int row, int col) {

	if (row >= matrix.length || col >= matrix[0].length) {
		return;
	}
	if (row == matrix.length - 1 && col == matrix[0].length - 1) {
		max = Math.max(min, max);
		return;
	}
	min = Math.min(matrix[row][col], min);
	helper(matrix, min, row + 1, col);
	helper(matrix, min, row, col + 1);

}

//Amazon phone count and say

public String encode (String str) {
	if (str == null || str.length() == 0) {
		return "";
	}
	if (str.length() == 1) {
		return str;
	}
	String result = "";
	int start = 0; 
	int i = 1;
	while (i < str.length()) {
		char pre = str.charAt(i - 1);
		char cur = str.charAt(i);
		if (pre != cur) {
			result += (i - start) + str.charAt(start) + "";
			start = i;
		}
		i++;
	}
	if (i - start >= 1) {
		result += (i - start) + str.charAt(start) + "";
	}
	return result;
}
public String decode (String str) {
	if (str == null || str.length() == 0) {
		return "";
	}
	String result = "";
	int start = 0;
	int i = 0;
	while (i < str.length()) {
		while (Character.isDigit(str.charAt(i) && i < str.length()) {
			i++;
		}
		if (i >= str.length()) {
			return "";
		}
		int count = Integer.parseInt(str.substring(start, i));
		char c = str.charAt(i);
		for (int j = 0; j < count; j++) {
			result += "" + c;
		}
		start = i + 1;
		i++;
	}
	return result;
}

//Amazon OA shortest job first
int waitTime = 0;
int curTime = 0;
int index = 0;
while (!heap.isEmpty() || index < len) {
	if (!heap.isEmpty()) {
		Process pro = heap.poll();
		waitTime += curTime - pro.cur;
		curTime += Math.min(pro.exe, q);

		while (index < len && req[index] <= curTime) {
			heap.offeR();
			index++;
		}
		if (cur.exe > q) {
			heap.offer(new);
		}
	} else {
		heap.offer(new Process(req[index], dur[index]));
		curTime = req[index++];
	}
}


public List<String> comb (String pwd, Map<Character, Character> map) {

		List<String> list = new ArrayList<>();
		if (pwd == null || pwd.length() == 0) {
			return list;
		}
		if (pwd.length() == 1) {
			char c = pwd.charAt(0);
			list.add("" + c);
			if (map.containsKey(c)) {
				list.add("" + map.get(c));
			}  

			return list;
		}
		List<String> left = new ArrayList<>();
		char c = pwd.charAt(0);
		left.add("" + c);
		if (map.containsKey(c)) {
			left.add("" + map.get(c));
		}  
		
		List<String> right = comb(pwd.substring(1, pwd.length()), map);
		
		for (int i = 0; i < left.size(); i++) {
			for (int j = 0; j < right.size(); j++) {
				list.add(left.get(i) + right.get(j) + "");
			}
		}

		return list;
	}



import java.util.*;

class Node {
	int val;
	List<Node> children;
	public Node (int val) {
		this.val = val;
		children = new ArrayList<>();
	}
}
class SumCount {
	int sum;
	int count;
	public SumCount (int sum, int count) {
		this.sum = sum;
		this.count = count;
	}
}
public class Company_Tree {
	private static double resAve = Double.MIN_VALUE;
	private static Node result;
	public static Node getHighAve (Node root) {
		if (root == null) {
			return null;
		}
		helper(root);
		return result;
	}
	public static SumCount helper (Node root) {
		if (root.children == null || root.children.size() == 0) {
			return new SumCount(root.val, 1);
		}
		int curSum = root.val;
		int curCnt = 1;
		for (Node node : root.children) {
			SumCount temp = helper(node);
			curSum += temp.sum;
			curCnt += temp.count;
		}
		double curAve = (double) curSum / curCnt;
		if (resAve < curAve) {
			resAve = curAve;
			result = root;
		}
		return new SumCount(curSum, curCnt);
	}
}






//maximum minmum path
int max = Integer.MIN_VALUE;
int rowNum;
int colNum;
public maxMinPath helper(int[][] matrix){
	rowNum = matrix.length;
	colNum = matrix[0].length;
	int min = Integer.MAX_VALUE;
	helper(matrix, min, 0, 0);
	return min;
}
public void helper (int[][] matrix, int min, int row, int col) {
	if (row >= rowNum || col >= colNum) {
		return;
	}
	if (rowNum - 1 == row && colNum - 1 == col) {
		min = Math.min(min, matrix[row][col]);
		max = Math.max(max, min);
		return;
	}
	min = Math.min(min, matrix[i][j]);
	helepr(matrix, min, row + 1, col);
	helper(matrix, min ,row, col + 1);
}

// amazon high five
class Result {
	int id;
	int val;
	public Result (int id, int val) {
		this.id = id;
		this.val = val;
	}
}
public static Map<Integer, Double> getHighFive(Result[] results){

	Map<Integer, Double> result = new HashMap<>();
	Map<Integer, PriorityQueue<Integer>> map = new HashMap<>();
	for (Result res : results) {
		if (map.containsKey(res.id)) {
			PriorityQueue<Integer> heap = map.get(res.id);
			heap.offer(res.val);
			if (heap.size() > 5) {
				heap.poll();
			}
			map.put(res.id, heap);
		} else {
			PriorityQueue<Integer> heap = new PriorityQueue<>(new Comparator<Integer> () {
				public int compare (Integer i1, Integer i2) {
					return i1 - i2;
				}
			});
			heap.offer(res.val);
			map.put(res.id, heap);
		}
	}
	for (Integer id : map.keySet()) {
		double sum = 0;
		while (!map.get(id).isEmpty) {
			sum += map.get(id).poll();
		}
		sum /= 5.0d;
		result.put(id, sum);
	}
	return result;
}


public Point[] Solution(Point[] array, Point origin, int k) {

	PriorityQueue<Point> heap = new PriorityQueue<>(k, new Comparator<Point>() {
		@Override
		public int compare (Point p1, Point p2) {
			return (int) getDistance(p2, origin) - getDistance(p1, origin); 
		}
	});

	for (Point p : array) {
		heap.offer(p);
		if (heap.size() > k) {
			heap.poll();
		}
	}
	Point[] result = new Point[k];
	int i = 0;
	while (!heap.isEmpty()) {
		result[i++] = heap.poll();
	}
	return result;
}

public double getDistance (Point p1, Point p2) {
	return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}
public int getAdsMaxProfit2(Ad[] ads, int time) {
    int[] profit = new int[time + 1];
    Arrays.sort(ads, new Comparator<Ad>() {
    	public int compare(Ad a1, Ad a2) {
        	if (a1.endTime == a2.endTime) {
                return a1.startTime - a2.startTime;
            }
        	return a1.endTime - a2.endTime;
        }
    });
    for (int i = 1; i <= time; i++) {
        profit[i] = profit[i - 1];
        for (int j = 1; j <= ads.length; j++) {
            if (ads[j - 1].endTime <= i) {  
                profit[i] = Math.max(profit[i], profit[ads[j - 1].startTime] + ads[j - 1].profit);-google 1point3acres
            } else {
                break; // others' endTime must be larger than i
            }
        }
    }
    return profit[time];
}


public int findDuplicate(int[] nums) {



}

public class Solution {
    public int longestConsecutive(TreeNode root) {
        if(root == null){
            return 0;
        }
        return findLongest(root, 0, root.val - 1);
    }
    
    private int findLongest(TreeNode root, int length, int preVal){
        if(root == null){
            return length;
        }
        // 判断当前是否连续
        int currLen = preVal + 1 == root.val ? length + 1 : 1;
        // 返回当前长度，左子树长度，和右子树长度中较大的那个
        return Math.max(currLen, Math.max(findLongest(root.left, currLen, root.val), findLongest(root.right, currLen, root.val)));  
    }
}


public void print (int n) {
	for (int i = 0; i < n; i++) {
		char[] arrs = new char[2 * n - 1];
		int mid = (2 * n - 1) / 2;
		for (int j = 0; j <= i; j++) {
			arrs[mid + j] = '*';
			arrs[mid - j] = '*'
		}
		System.out.println(arrs);
	}
}

public TreeNode UpsideDownBinaryTree(TreeNode root) {

	if (root == null) {
		return root;
	}
	List<TreeNode> list = new ArrayList<>();
	list.add(null);
	helper(root, list);
	return list.get(0);
}
public TreeNode (TreeNode root, List<TreeNode> list) {
	if (root.left == null) {
		list.set(0, root);
		return root;
	}
	TreeNdoe newRoot = helper(root.left, list);
	newRoot.left = root.right;
	newRoot.right = root;
	return newRoot.right;
}

//how many times to fill string into matrix

public int fillMatrix (String[] text, int[][] matrix) {
	
	int colNum = matrix[0].length;
	int rowNum = matrix.length;

	int count = 0;
	int curPos = 0;
	int result = 0;
	int i = 0;
	while (count < rowNum) {
		if (text[textIndex].length() > colNum) {
			return 0;
		}
		// change line
		// do not change line
		if (curPos + text[i].length() >=  colNum) {
			count += 1;
			curPos = text[i].length() - 1;
		} 

		if (count < rowNum) {
			if (i == text.length - 1) {
				result += 1;
				i = 0;
			} else {
				i += 1;
			}
		}
	}
	return result;
}


//maze
boolean result;
    public boolean maze(int[][] matrix) {

        if (matrix[0][0] == 9) {
            return true;
        }
        result = false;
        
        helper(matrix, 0, 0);
                
        return result;
    }
    public void helper (int[][] matrix, int row, int col) {
        if (row < 0 || row >= matrix.length || col < 0 
            || col >= matrix[0].length || (matrix[row][col] != 1 && matrix[row][col] != 9)) {
            return;
        }
        if (matrix[row][col] == 9) {
            result = true;
            return;
        }
        int temp = matrix[row][col];
        int[][] dir = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (int i = 0; i < 4; i++) {
            int x = dir[i][0] + row;
            int y = dir[i][1] + col;
            matrix[row][col] = 3;
            helper(matrix, x, y);
            matrix[row][col] = temp;
        }
    }


// t2 is subtree of t1
public static int check1(TreeNode t1, TreeNode t2) {

	if (t1 == null) {
		return false;
	}
	if (t2 == null) {
		return true;
	}
	if (helper(t1, t2)) {
		return true;
	}
	if (helper(t1.left, t2) || helper(t1.right, t2)) {
		return true;
	}
	return false;
}
public boolean helper (TreeNode root, TreeNode node) {
	if (root == null || node == null) {
		return root == node;
	}
	if (root.val != node.val) {
		return false;
	}
	return helper(root.left, node.left) && helper (root.right, node.right);
}

// word pattern ii
Map<Character, String> map;
Set<String> set;
boolean result;
public boolean wordPatternMatch(String pattern, String str) {
		// valid match
		map = new HashMap<>();
		set = new HashSet<>();
		result = false;
		helper(pattern, 0, str, 0);
		return result;
}
public boolean helper (String pattern, int pStart, String str, int sStart) {
	if (pStart == pattern.length() && sStart == str.length()) {
		result = true;
		return;
	}
	if (pStart >= pattern.length() || sStart >= str.length()) {
		return;
	}
	char c = pattern.charAt(pStart);
	for (int cut = sStart + 1; cut <= str.length(); cut++) {
		String cur = str.susbtring(sStart, cut);
		if (!set.contains(cur) && !map.containsKey(c)) {
			map.put(c, cur);
			set.add(cur);
			helper(pattern, pStart + 1, str, cut);
			set.remove(cur);
			map.remove(c);
		} else if (map.containsKey(c) && map.get(c).equals(cur)) {
			helper(pattern, pStart + 1, str, cut);
		}
	}
}

String[] patterns = {"", "", "abc", "def", "ghi", ""};
public List<String> letterCombinations(String digits) {

	List<String> result = new ArrayList<>();
	helper(result, digits, 0, new StringBuilder());
	return result;
}
public void helper (List<String> result, String digits, int index, StringBuilder sb) {
	if (sb.length() >= digits.length()) {
		result.add(sb.toString());
		return;
	}
	
		String letters = patterns[digits.charAt(index)];
		for (char c : letters.toCharArray()) {
			sb.append(c);
			helper(result, digits, index + 1, sb);
			sb.deleteCharAt(sb.length() - 1);
		}
	
}



public class TicTacToe {
	int[] hor;
	int[] ver;
	int diaL;
	int diaR;
	int n;
	TicTacToe (int n) {
		this.n = n;
		this.hor = new int[n];
		this.ver = new int[n];
		this.diaL = 0;
		this.diaR = 0;
	}
	public int move (int row, int col, int player) {
		int count = player == 1 ? 1 : -1;
		
		hor[row] += count;
		ver[col] += count;

		if (row == col) {
			diaL += count;
		}
		if (row + col = n - 1) {
			diaR += count;
		}

		if (Math.abs(hor[row] == n) || Math.abs(ver[col] == n || Math.abs(diaL) == n || Math.abs(diaR) == n)) {
			return count > 0 ? 1 : 2;
		}
		return 0;
	}

}


abcabcaaaaaaaaaaaaabc
abcccc
a1b1c4
public String stringCompression (String input) {
	if (input == null || input.length() == 0) {
		return;
	}
	StringBuilder sb = new StringBuilder();
	if (input.length() == 1) {
		sb.append(input[0]);
		sb.append(1);
		return sb.toString();
	}
	char[] chars = input.toCharArray(); 
	int start = 0;
	int i = 1; 
	while (i < chars.length) {
		if (input.charAt(i - 1) != input.charAt(i)) {
			sb.append(input.charAt(start));
			sb.append(i - start);
			start = i;
		}
		i += 1;
	}
	if (start <= input.length()) {
		sb.append(input.charAt(start));
		sb.append(i - start);
	}

	return sb.toString();
}




List<String[]> = list = new ArrayList<>();
list.add(new String[]{"William", "Ryan"});
list.add(new String[]{"Chirley", "Ryan"});
list.add(new String[]{"Ryan", "Bob"});
list.add(new String[]{"Bob", "Daniel"});
list.add(new String[]{"Kilsey", "Bob"});
list.add(new String[]{"Daniel", "null"});
list.add(new String[]{"Miao", "null"});




public List<String> companyLevel (List<String[]> levels) {

	Map<String, List<String>> map = new HashMap<>();
	for (String[] level : levels) {
		List<String> cur = map.get(level[1]);
		if (cur == null) {
			cur = new ArrayList<>();
		}
		cur.add(level[0]);
		map.put(level[1], cur);
	}
	List<String> result = new ArrayList<>();
	helper (result, map, "null");
	return result;
}
public void helper (List<String> list, Map<String, List<String>> map, String str) {

	if (map.get(str) == null) {
		return;
	}
	for (String s : map.get(str)) {
		list.add(s);
		helper(list, map, s);
	}

}





public calss HashTableEntyr <K, V> {
	private K key;
	private V value;
	private int hash;
	private HashTableEnty<K, V> next;

}
public class HashMap<K, V> {
	HashTableEnty[] tab;
	public HashMap() {
		tab = new HashTableEnty[default_size];
	}
	public V get (Object key) {
		int hash = key.hashCode();
		HashTableEnty<K, V> cur = tab[hash];
		while (cur != null && cur.key != key) {
			cur = cur.next;
		}
		return cur == null ? null : cur.val;
	}
	public void put (K, V) {
		int hash = K.hashCode();
		if (tab[hash].K == K) {
			HashTableEnty<K, V> cur = new HashTableEnty<K, V>();
		}
	}
}



enum AttackResponse {
        HIT, MISS, SUNK, END
}

    class Ship {
        private List<Coordination> parts;
        GameBoard board;

        void addPart(Coordination coor) {
            parts.add(coor);
        }

        void removePart(Coordination coor) {
            parts.remove(coor);
        }
        boolean hasSunk() {
            return parts == null || parts.size() == 0;
        }
    }

    class Coordination {
        private int x;
        private int y;
    }

    class Player {
        int id;
        GameBoard board;

        AttackResponse Attack(GameBoard rivalBoard, Coordination shot) {
            if(!rivalBoard.checkCoordinationInfo(shot)) {
                return AttackResponse.MISS;
            }
            else {
                Ship hittedShip = rivalBoard.getRelatedShip(shot);
                rivalBoard.removeRelation(shot);
                if(!rivalBoard.isShipSunk(hittedShip)) {
                    return AttackResponse.HIT;
                }
                else if(!rivalBoard.isGameEnd()){
                    return AttackResponse.SUNK;
                }
                else {
                    return AttackResponse.END;
                }
            }
        }

    }

    class GameBoard {
        //List<Ship> ships;
        //boolean[][] hasPart;
        private HashMap<Coordination, Ship> relations;
        private Player player;

        boolean checkCoordinationInfo(Coordination coor) {
            return relations.containsKey(coor);
        }

        void removeRelation(Coordination coor) {
            relations.remove(coor);
        }

        Ship getRelatedShip(Coordination coor) throws Exception {
            if(relations.containsKey(coor)){
                return relations.get(coor);
            }
            else {                
                throw new Exception(&quot;wrong coordination!&quot;);
            }
        }
        boolean isShipSunk(Ship ship) {
            return !relations.containsValue(ship);
        }

        boolean isGameEnd() {
            return relations == null || relations.size() == 0;
        }
    }





public enum VehicleSize {
	Car, 
	Bus, 
	MotoCycle,
}

abstract class Vehicle {
	String plateNumber;
	VehicleSize size;
	int spaceNeed;
	List<ParkingSpot> list = new ArrayList<>();
	public int getSpotsNeed () {
		return spaceNeed;
	}
	public VehicleSize getVehicleSize () {
		return size;
	}

	public void enterIn(ParkingLot spot) {
		list.add(spot);
	}
	public void moveOut (){
		for (int i = 0; i < list.size(); i++){
			list.get(i).removeVehicle();
		}
		list.clear();
	}
	public abstract boolean canFitInSpot (ParkingSpot spot);
}

class Bus extends Vehicle{
	public Bus () {
		spaceNeed = 5;
		size = VehicleSize.Bus;	
	}
	public boolean canFitInSpot(ParkingSpot spot) {
		return spot.getSize() == VehicleSize.Bus;
	}
}
class Car extends Vehicle {
	public Car () {
		spaceNeed = 1;
		size = VehicleSize.Car;	
	}
	public boolean canFitInSpot(ParkingSpot spot) {
		return spot.getSize() == VehicleSize.Bus || spot.getSize() == vehicleSize.Car;
	}

}
class MotoCycle extends Vehicle {
	public MotoCycle () {
		spaceNeed = 1;
		size = VehicleSize.MotoCycle;
	}
	public boolean canFitInSpot(){
		return true;
	}
}


public class ParkingSpot {
	Vehicle vehicle;
	VehicleSize size;
	int row;
	Level level;
	int spotNumber;

	Date move_in;
	Date move_out;

	ParkingSpot (Level level, int row, int spotNumber, VehicleSize size) {
		this.level = level;
		this.row = row;
		this.spotNumber = spotNumber;
		this.size = size;
	}

	public VehicleSize getSize () {
		return size;
	}
	public int getrow () {
		return row;
	}
	public int getSpotNumber (){
		return spotNumber;
	}

	public void removeVehicle () {
		level.spotFreed();
		vehicle = null;
	}
	public boolean park (Vehicle vehicle) {
		if (!canFitVehicle(vehicle)) {
			return false;
		}
		vehicle = vehicle;
		vehicle.enterIn(this);
		return true;
	}
	public boolean isAvailable () {
		return vehicle == null;
	}
	public boolean canFitVehicle (Vehicle vehicle) {
		 return isAvailable() && vehicle.canFitInSpot(this);
	}
}


class ParkingLot {
	Level[] levels;
	int num_Levels;

	public ParkingLot (int n, int num_rows, int spots_pre_row) {
		this.num_Levels = n;
		levels = new Level[n];
		for (int i = 0; i < n; i++) {
			levels[i] = new Level(i, num_rows, spots_pre_row);
		}
	}
	public boolean parkVehicle (Vehicle vehicle) {
		for (int i = 0; i < levels.length; i++) {
			if (levels[i].parkVehicle(vechile)) {
				return true;
			}
		}
		return false;
	}
	public void unParkVehicle (vehicle) {
		vehicle.moveOut();
	}

}

class Level {
	int floor;
	ParkingSpot[] spots;
	int availableSpot = 0;
	int spots_per_row ;

	public Level (int floor, int num_row, int spots_per_row) {
		this.floor = floor;
		int SPOTS_PER_ROW = spots_per_row;
		int spot_Index = 0;
		spots = new ParkingSpot[num_row * spots_per_row];

		//init size for each spot in array spots
		for (int row = 0; row < num_rows; row++) {
			for (int spot = 0; spot < spots_per_row / 4; spot++) {
				VehicleSize size = VehicleSize.MotoCycle;
				spots[spot_Index] = new ParkingSpot(this, row, spot_Index, size);
				spot_Index += 1;
			}
			for (int spot = spots_per_row / 4; spot < spots_per_row / 4 * 3; ++spot) {
                VehicleSize sz = VehicleSize.Compact;
                spots[numberSpots] = new ParkingSpot(this, row, numberSpots, sz);
                numberSpots ++;
            }
            for (int spot = spots_per_row / 4 * 3; spot < spots_per_row; ++spot) {
                VehicleSize sz = VehicleSize.Large;
                spots[numberSpots] = new ParkingSpot(this, row, numberSpots, sz);
                numberSpots ++;
            }
		}
		availableSpot = numberSpots;
	
	}
	public int findAvailableSpots (Vehicle vechile) {
		int spotsNeeded = vehicle.getSpotsNeed();
		int lastRow = -1;
		int spotsFound = 0;
		for (int i = 0; i < spts.length; i++) {
			ParkingSpot spot = spots[i];
			if (lastrow != spot.getRow()) {
				spotsFoun = 0;
				lastRow = spot.getRow();
			}
			if (spot.canFitVehicle(vehicle)) {
				spotsFound += 1;
			} else {
				spotsFound = 0;
			}
			if (spotsFound == spotsNeeded) {
				return i - (spotsNeeded - 1);
			}
		}
		return -1;
	}
	public boolean parkStartingAtSpot (int spotNumber, Vehicle vehicle) {

	}
	public void spotFreed () {
		availableSpot += 1;
	}
	public int availableSpot () {
		return availableSpot;
	}
	
}






















Set<Character> notValid = new HashSet<>();
Map<Character, Integer> map;
String target;
int len;
int count;
wordGuess (String target) {
	this.count = 0;
	this.len = target.length();
	this.target = target;
	map = new HashMap<>();
	for (int i = 0;i < target.length(); i++) {
		map.put(target.charAt(i), i);
	}
}

boolean couldTry = false;
StringBuilder sb = new StringBuilder();
if (char c : word.toCharArray()) {
	if (!notValid.contains(c)) {
		sb.append(c);
		couldTry = true;;
	}
}

if (couldTrey) {
	String wordGuess(sb.toString());

}

public String wordGuess (String word) {
	for (char c : word.toCharArray()) {
		if (notValid.contains(c)) {
			continue;
		}
		if (map.containsKey(c)) {
			result[map.get(c)] = c;
			count += 1;
			notValid.add(c);
		}
	}
	return new String(result);
}



class Node{
	int val;
	boolean isVisible;
	Node(val) {
		this.val = val;
		isVisible = false;
	}
}

int mWidth;
int mHeight
int mines;
Node[][] minesField;

public void placeMines () {
	minesField = new int[mWidth][mHeight];
	int placeMines = 0;
	Random rand = new Random();
	int total = mWidth * mHeight;
	while (mines < placeMines) {
		int r = rand.nextInt(total);
		
		int row = r / mWidth;
		int col = r % mHeight;
		if (minesField[row][col] == 9) {
			continue;
		}
		
		placeMines += 1;
		for (int i = Math.max(0, row - 1); i <= Math.min(row + 1, mWidth - 1); i++) {
			for (int j = Math.max(0, col - 1); j <= Math.min(mHeight - 1, col + 1); j++) {
				if (i == row && j == col) {
					minesFied[i][j] = 9;
				} else {
					minesFied[i][j] += 1;
				}
			}
		}		

	}

}
public boolean onClick (int row, int col) {
	if () {
		return false;
	}
	if (minesField[row][col].isVisible) {
		return false;
	}

	minesField[row][col].isVisible = true;
	if (minesField[row][col].val == 9) {
		return true;
	}
	if (minesField[row][col] != 0) {
		return false;
	}

	onClick(row - 1, col);

	return false;

}








public boolean synonym queries (List<List<String>> input) {
	Map<String, Set<String>> map = new HashMap<>();

	for (List<String> list : input) {
		String[] from = list.get(0).split("\\s+"); 
		String[] to = list.get(1).split("\\s+");

		int i = 0;
		int j = 0;
		while (i < from.length && j < to.length) {
			String pre = from[i];
			String next = to[i];
			if (map.containsKey(pre)) {
				if (map.get(pre).contains(next)) {
					i += 1;
					j += 1;
				} else {
					return false;
				}
			} else {
				if (map.containsKey(next)) {
					if (map.get(next) == pre) {
						i += 1;
						j += 1;
					} else {
						return false;
					}
				} else {
					i += 1;
					j += 1;
				}
			}
		}
	}
	return true;
}


public List<int[]> smallestRect (List<int[]> points) {
	TreeMap<Integer, List<int[]>> map = new HashMap<>();
	for (int[] point : points) {
		List<int[]> list = map.get(point[1]);
		if (list == null) {
			list = new ArrayList<>();
		}
		list.add(point);
		map.put(point[1], list);
	}
	List<int[]> pre = new ArrayList<>();
	for (Map.Entry<Integer, List<int[]>> entry : map.entrySet()) {
		int height = entry.getKey();
		List<int[]> list = entry.getValue();
		if (list.size() <= 1) {
			continue;
		} else {
			if (pre.size() == 0) {
				pre = list;
			} else {
				for () {

				}
			}
		}
	}

	int pre = heights.poll();
	while (!heights.isEmpty()) {

	}
	return result;
} 



public String decompressed (String input) {
	if (input == null || input.length() == 0) {
		return "";
	}
	String result = "";
	StringBuilder sb = new StringBuilder();
	int start = 0;
	Stack<Integer> stack = new Stack<>();
	while (i < input || !stack.isEmpty()) {
		char c = input.charAt(i);
		if (c == '[') {
			int = Integer.parseInt(sb.toString());
			sb = new StringBuilder();
			i += 1;
		} else if (c == ']') {
 			int count = stack.pop();
 			
 			for (int i = 0; i < count - 1; i++) {
 				sb.append(sb);
 			}
 			result += sb.toString();
  			i += 1;
 			start = i + 1;
		} else {
			sb.append(c);
			i += 1;
		}
	}
	if (sb.length() > 0) {
		result += sb.toString();
	}
	return result;
}


public int findKth (int[] A, int lenA, int startA, int[] B, int lenB, int startB,int k) {

	if (lenA > lenB) {
		return findKth(B, lenB, startB, A, lenA, startA, k);
	}

	if (lenA == 0) {
		return B[startB + k - 1];
	}
	if (k == 1) {
		return Math.min(A[startA], B[startB]);
	}

	int p1 = Math.min(lenA, k / 2);
	int p2 = k - p1;

	if (A[startA + p1 - 1] < B[startB + p2 - 1]) {
		return helper(A, lenA - p1, startA + p1, B, lenB, k - p1);
	} else if (A[startA + p1 - 1] > B[startB + p2 - 1]) {
		return helper(A, lenA, startA, B, lenB + p2, k - p2);
	} else {
		return A[startA + p1 - 1];
	}

}


public char mostFrequentChar (String str) {
	int[] chars = new int[26];
	int max = Integer.MIN_VALUE;
	int count = 1;
	for (int i = 0; i < str.length(); i++) {
		chars[str.charAt(i) - 'a'] += 1;
		if (chars[str.charAt(i) - 'a'] > max) {
			max = chars[str.charAt(i) - 'a'];
			count = 1;
		} else if (chars[str.charAt(i) - 'a'] == max) {
			count += 1;
		}
	}
	if (count == 1;) {
		for (int i = 0; i < 26; i++) {
			if (chars[i] == max) {
				return (char)(i + 'a');
			}
		}
	} else {
		for (int i = 0; i < str.length(); i++) {
			if (chars[i] == count) {
				return (char)(i + 'a');
			}
		}
	}
}



public List<String> addOperators(String num, int target) {

	String symbol = "+-*";
	List<String> result = new ArrayList<>();
	StringBuilder sb = new StringBuilder();
	helper(num, target, result, symbol, sb, 0);
	return result;
}
public void helper () {

}






class Level {

	int floor;
	ParkingSpot[] spots;
	int availableSpot = 0;
	final int spots_per_row = 10;

	public Level (int flor, int numberSpots) {

	}
	public int availableSpot () {
		return availableSpot;
	}
	public boolean parkVehicle (Vehicle vehicle) {

	}
	boolean parkStartingAtSpot (int num, Vehicle v) {

	}
	int findAvailableSpots (Vehicle vehicle) {

	}
	void spotFreed () {
		availableSpot += 1;
	}

	
}
public class ParkingLot {
	Level[] levels;
	final int num_Levels = 5;

	public ParkingLot () {

	}
	public boolean parkVehicle (Vehicle vehicle) {

	}

}
class ParkingSpot {
	int row;
	int parkingSpotId;
	Level level;
	Vehicle vehicle;
	VehicleSize vehicleSize;

	public ParkingSpot (Level level, int row, int spotId, VehicleSize size) {

	}
	public boolean isAvailable () {
		return vehicle == null;
	}
	public boolean canFitVehicle (Vehcile vehicle) {

	}
	public boolean park (Vehicle vehicle) {

	}
	public int getRow () {
		return row;
	}
	public int getSpotId () {
		return parkingSpotId;
	}
	public void removeVehicle () {

	}

}



public class enum VehicleSize {
	Bus, Compact, MotoCycle
}

public abstract class Vehicle {
	List<ParkingSpot> parkingSpots; = new ArrayList<>();
	String plateNumber;
	VehicleSize vehicleSize;
	int parkingSpotNeeded;
	
	public int getParkingSpotNeeded () {
		return parkingSpotNeeded;
	}

	public VehicleSize getVehicleSize () {
		return vehicleSize;
	}
	public void parkInSpot (ParkingSpot s) {
		parkingSpots.add(s);
	}
	public void moveOutSpt () {
		parkingSpots = new ArrayList<>();
	}
	public abstract boolean canFitInSpot (ParkingSpot spot);
}
public class MotoCycle extends Vehicle {
	public MotoCycle () {
		parkingSpotNeeded = 1;
		vehicleSize = VehicleSize.MotoCycle;
	}
}
class Bus extends Vehicle {
	public Bus () {
		parkingSpotNeeded = 1;
		vehicleSize = VehicleSize.Bus;
	}
}
class Compact extends Vehicle {
	public Compact () {
		parkingSpotNeeded = 5;
		vehicleSize = VehicleSize.Bus;
	}
}

public int depthSum(List<NestedInteger> nestedList) {

	Map<Integer, Map<Integer, Integer>> map = new HashMap<>();

	for (NestedInteger n : nestedList) {
		List<> = NextedInteger.getList().size();
		map.put()
	}
}



public boolean isIdentical (TreeNode root1, TreeNode root2) {
	if (root1 == null && root2 == null) {
		return true;
	}

	if (root1 == null || root2 == null) {
		return false;
	}
	if (root1.val != root2.val) {
		return false;
	}
	return isIdentical(root1.left, root2.left) && (isIdentical(root1.right, root2.right))
}



String result = "";
public String longestPalindrome(String s) {

	for (int i = 0; i < s.length(); i++) {
		helper(s, i, 0);
		helper (s, i, 1); 
	}
	return result;
}
public void helper (String s, int index, int offset) {
	int left = index;
	int right = index + offset;
	while (0 <= left && right < s.length() && s.charAt(left) == s.charAt(right)) {
		left--;
		right++;
	}
	String curLongest = s.susbtring(left + 1, right);
	if (curLongest.length() > result.length()) {
		result = curLongest;
	}
}





public int numDecodings(String s) {

	if (s.charAt(0) == '0') {
		return 0;
	}
	int first = 1;
	int second = 1;

	for (int i = 1; i < s.length(); i++) {
		char c = s.charAt(i);
		if (c == '0') {
			if ('1' == s.charAt(i - 1) || s.charAt(i - 1) == '2') {
				second = first;
			} else {
				return 0;
			}
		} else {
			
			if (s.charAt(i - 1) == '1' || s.charAt(i - 1) == '2' && s.charAt(i) - '0' <= 6) {
				second = first + second;
				first = second - first;
			}  else {
				first = second;
			}
		}
	}
	return arrs[arrs.length - 1];
}





public int minCostII(int[][] costs) {

	int preMin = 0;
	int preSecMin = 0;
	int preMinPos = -1;

	for (int i = 0; i < costs.length; i++) {
		int curMin = Integer.MAX_VALUE;
		int curSecMin = Integer.MAX_VALUE;
		int curMinPos = -1;
		for (int j = 0; j < costs[0].length; j++) {

			costs[i][j] += preMinPos == j ? preSecMin : preMin;

			if (costs[i][j] < curMin) {
				if (curSecMin != Integer.MAX_VALUE) {
					curSecMin = costs[i][j]; 
				}
				curMin = costs[i][j];
				curMinPos = j;

			} else if (costs[i][j] < curSecMin) {
				curSecMin = costs[i][j];
			}
		}
		preMin = curMin;
		preSecMin = curSecMin;
		preMinPos = curMinPos;
	}
	return preMin;

}



//Random Maximum
public int findMax(int[] arr){
	int result = -1;
	int max = Integer.MIN_VALUE;
	int count = 1;
	for (int i = 0; i < arr.length; i++) {
		if (arr[i] == max) {
			count += 1;
			int pos = new Random.nextInt(count);
			if (pos == 0) {
				result = i;
			}
		} else if (arr[i] > max || max = Integer.MIN_VALUE) {			
				max = arr[i];
				result = i;
				count = 1;
		}
	}
	return result;
}

//输出连续字符最多的
public List<Character> longestConsecutive (String str) {
	List<Character> list = new ArrayList<>();
	if (str == null || str.length() == 0) {
		return result;
	}
	char pre = str.charAt(0);
	int count = 1;
	int preMaxCount = -1;
	for (int i = 1; i < str.length(); i++ ) {
		if (pre == str.charAt(i)) {
			count += 1;
		} else {
			if (count >= preMaxCount) {
				helper (pre, count, preMaxCount, list);
				if (pre != ' ') {
					preMaxCount = count;
				}
			}
			pre = str.charAt(i);
			count = 1;
		}
	}
	return list;
}
public void add (List<Character> list, int count, int preMaxCount, char cur) {
	if (cur == ' ') {
		return;
	}
	if (count > preMaxCount) {
		list.clear();
		list.add(cur);
	} else if (count == preMaxCount) {
		list.add(cur);
	}
}



// mulitiply strings
public String multiply(String num1, String num2) {
	int len1 = num1.length();
	int len2 = num2.length();
	


	long[] arrs = new long[len1 + len2 - 1];

	for (int i = num2.length() - 1; i >= 0; i--) {
		for (int j = num1.length() - 1; j >= 0; j--) {

			arrs[i + j] += (num1.charAt(i) - 'a') * (num2.charAt(j) - 'a');
		}
	}
	StringBuilder sb = new StringBuilder();
	int carry = 0;
	for (int i = arrs.length - 1; i >= 0; i--) {
		sb.insert(0, (arrs[i] + carry) % 10);
		carry = arrs[i] / 10;
	}
	while (sb.charAt(0) == '0' && sb.length() > 1) {
		sb.deletCharAt(0);
	}

	return sb.toString();
}


//longest decreasing path(滑雪)
int[][] dp;
public int getLongestPath(int[][] matrix){
	dp = new int[matrix.length][matrix[0].length];
	int max = 0;
	for (int i = 0; i < matrix.length; i++) {
		for (int j = 0; j < matrix[0].length; j++) {
			dp[i][j] = helper (matrix, i, j);
			if (dp[i][j] > max) {
				max = dp[i][j];
			}
		}
	}
	return max;
}
public int helper (int[][] matrix, int row, int col) {
	int length = 1;
        // 递归上下左右
        if(i > 0 && m[i - 1][j] < m[i][j]){
            length = Math.max(dfs(i - 1, j, m) + 1, length);
        }
        if(j > 0 && m[i][j - 1] < m[i][j]){
            length = Math.max(dfs(i, j - 1, m) + 1, length);
        }
        if(i < m.length - 1 && m[i + 1][j] < m[i][j]){
            length = Math.max(dfs(i + 1, j, m) + 1, length);
        }
        if(j < m[0].length - 1 && m[i][j + 1] < m[i][j]){
            length = Math.max(dfs(i, j + 1, m) + 1, length);
        }
        dp[i][j] = length;
        return length;
}



//Facebook IntFileIterator
class IntFileIterator {
  boolean hasNext();
  int next();
}

class{
  public boolean isDistanceZeroOrOne(IntFileIterator a, IntFileIterator b)；
  	int diffCount = 0;
  	while (a.hasNext() && b.hasNext()) {
  		int curA = a.next();
  		int curB = b.next();
  		if (curA != curB) {
  			int preA = curA;
  			int preB = curB;
  			if (a.hasNext() && b.hasNext()) {
  				curA = a.next();
  				curB = b.next();
  			
	  			if (curA != preB && curB != preA) {
	  				if (curA != curB) {
	  					return false;
	  				}
	  				return isSame(a, b);
	  			}
  	  		
	  			if (curA == preB && curB == preA) {
	  				return false;
	  			}

	  			if (curA == preB) {
	  				if (curA != curB) {
	  					return isAdd(a, b, curB);
	  				}
	  				return isAddOrChange(a, b, curB);
	  			}

  	  		}
  		}
  	} 
  	
}



public Interval outputAsOrder(List<Interval> ints) {

	PriorityQueue<Interval> heap = new PriorityQueue<>(new Comparator<Interval>(){
		public int compare (Interval i1, Interval i2) {
			return i1.start - i2.start;
		}
	});
	Interval result = new Interval(0, 0);
	for (Interval i : ints) {
		heap.offer(i);
	}
	Interval pre = ints.poll();
	int count = 1;
	while (!heap.isEmpty()) {
		Interval cur = heap.poll();
		if (cur.start > pre.end) {
			pre = cur;
		} else {
			pre.start = cur.start;
			pre.end = Math.max(cur.end, pre.end);
		}
	}
}





class Contact {
	String name;
	List<String> email;
	Contac() {
		this.name = "";
		this.email = new ArrayList<>();
	}
}
public List<List<Contact>> (List<Contact> input) {
	Map<String, List<Integer>> map = new HashMap<>(); 
	int n = input.size();
	for (int i = 0; i < n; i++) {
		for (String e : Contact.email) {
			List<Integer> list;
			if (!map.containsKey(e)) {
				list = new ArrayList<>();
				list.add(i);
			} else {
				list = map.get(e));
				list.add(i);
			}
			map.put(e, list);
		}
	}

	unionFind uf = new unionFind(n);

	for (List<Integer> list : map.keySet()) {
		for (int i = 0; i < list.size() - 1; i++) {
			uf.unionFind(list.get(i), list.get(i + 1));
		}
	}
	Map<Integer, List<Integer>> groups = new HashMap<>();
	for (int i = 0; i < n; i++) {
		List<Integer> list;
		int cur = uf.find(i);
			if (!groups.containsKey(cur)) {
				list = new ArrayList<>();
				list.add(i);
			} else {
				list = groups.get(cur));
				list.add(i);
			}
			groups.put(cur, list);
	}
	List<List<Contact>> result = new ArrayList<>();
	for (List<Integer> list : groups.keySet()) {
		List<Contact> r = new ArrayList<>();
		for (int i : list) {
			r.add(input.get(i));
		}
		result.add(r);
	}
	return result;
}

class unionFind {
	int[] parent;
	public unionFind (int num_node) {
		for (int i = 0; i < num_node; i++) {
			parent[i] = i;
		}
	}
	public int find (int num) {
		if (parent[num] == numm) {
			return num;
		}
		parent[num] = find(parent[num]);
		return parent[num];
	}
	public void union (int num1, int num2) {
		if (find(num1) != find(num2)) {
			parent[num1] = num2;
		}
	}

}
class Contact {
	String name;
	List<String> emails;
}

int[] parent;
public boolean validTree (int n, int[][] edges) {
	parent = new int[n];
	for (int i = 0; i < n; i++) {
		parent[i] = i;
	}
	for (int[] edge : edges) {
		if (find(edge[0]) == find(edge[1])) {
			return false;
		}
		union(edge[0], edge[1]);
	}
	return true;
}
public int find (int pos) {
	if (parent[pos] == pos) {
		return pos;
	}
	parent[n] = find(parent[pos]);
	return parent[n];
}
public void union (int n1, int n2) {
	int r1 = find(n1);
	int r2 = find(n2);
	if (r1 != r2) {
		parent[r1] = r2;
	}
}





public int longestConsecutive(TreeNode root) {

	if (root == null) {
		return 0;
	}
	
	return helper(root, root.val - 1);
}
public int helper (TreeNode root, int preVal) {
	if (root == null) {
		return 0;
	}
	if (root.left == null && root.right == null) {
		return 1;
	}
	int left = helper (root.left, root.val);
	int right = helper (root.right, root.val);
	if (root.left.val == root.val + 1 && root.left != null) {
		left += 1;
	} else if (root.right.val == root.val + 1 && root.right != null) {
		right += 1;
	}
	return Math.max(left, right);
}


public static int longestConsecutive(int[] num) {
	if (num.length == 0 || num == null) {
		return 0
	}
	if (num.length == 1) {
		return 1;
	}
	Set<Integer> set = new HashSet<>();
	for (int n : num) {
		set.add(n);
	}
	int max = 0;
	for (int i = 0; i < num.length; i++) {
		//above
		int count = 0;
		int cur = num[i];
		int temp = cur;
		while (set.contains(temp)) {
			count += 1;
			set.remove(temp);
			temp = nums[i] + 1;
		}
		temp = cur - 1;
		while (set.contains(temp)) {
			count += 1;
			set.remove(temp);
			temp -= 1;
		}
		max = Math.max(max, count);
		//below
	}
	return max;
}


public String sumUp (String str1, String str2) {

	int index1 = str1.length() - 1;
	int index2 = str2.length() - 1;
	int carry = 0;
	StringBuilder sb = new StringBuilder();
	while (index1 > 0 && index2 > 0) {
		int cur1 = str1.charAt(i) - '0';
		int cur2 = str2.charAt(j) - '0';

		int sum = cur1 + cur2 + carry;
		sb.append(0, sum % 10);
		carry = sum / 10;
	}
	if (carry == 1) {
		sb.append("1");
	}
	return sb.toString();
}


List<Integer> merge (List<List<Integer>> list) {
	List<Iterator<Integer>> its = new ArrayList<>();
	for (List<Integer> l : list) {
		if (l.size() != 0) {
			its.add(l.iterator());
		}
	}
	List<Integer> result = new ArrayList<>();
	helper (its, result);
	return result;
}
public void helper (List<Iterator<Integer>> its, List<Integer> result) {

	int turn = 0;
	while (its.size() != 0) {
		int pos = turn % its.size();
		Iterator<Integer> cur = its.get(pos);
		result.add(cur.next());
		if (!cur.hasNext()) {
			its.remove(turn % its.size());
			turn -= 1;
		}
		turn += 1;
	}
	return result;
}




public List<Integer> numOfSqure (int num) {
	List<Integer> result = new ArrayList<>();
	if (num == 1) {
		result.add(1)
		return result;
	}
	while (num > 0) {
		int cur = helper(1, num, num);
		result.add(cur);
		num -= cur * cur;
	}
	return result;
}
public int firsSmaller (int start, int end, int target) {
	while (start + 1 < end) {
		int mid = start + (end - start) / 2;
		if (mid * mid < target) {
			start = mid;
		} else if (mid * mid > target) {
			end = mid;
		} else {
			return mid;
		}
	}
	if (start * start < target) {
		return start;
	}
	return end;
}


public List<List<Integer>> pathSum(TreeNode root, int sum) {
	List<List<Integer>> result = new ArrayList<>();
	helper (root, sum, result, new ArrayList<>());
	return result;
}
public void helepr (TreeNode root, int sum, List<List<Integer>> result, List<Integer> list) {
	if (root == null) {
		return;
	}
	if (root.left == null && root.right == null) {
		if (root.val == sum) {
			list.add(root);
			result.add(new ArrayList<>(list));
		}
		return;
	}
	if (root.left != null) {
		list.add(root.val);
		helper(root.left, sum - root.val, result, list);
		list.remove(list.size() - 1);
	}
	if (root.right != null) {
		list.add(root.val);
		helper(root.right, sum - root.val, result, list);
		list.remove(list.size() - 1);
	}
}



public void setZeroes(int[][] matrix) {

	boolean top = false;
	boolean left = false;
	for (int j = 0; j < matrix[0].length; j++) {
		if (matrix[0][j] == 1) {
			top = true;
			break;
		}
	}
	for (int i = 0; i < matrix.length; i++) {
		if (matrix[i][0] == 1) {
			left = true;
			break;
		}
	}


	for (int i = 1; i < matrix.length; i++) {
		for (int j = 1; j < matrix[0].length; j++) {
			if (matrix[i][j] == 0) {
				matrix[i][0] = 0;
				matrix[0][j] = 0;
			}
		}
	}

	for (int i = 0; i < matrix.length; i++) {

	}

	for (int j = 0; j < matrix[0].length; j++) {

	}
	if (top) {
		for (int j= 0) {

		}
	}
	if (left) {

	}
	return;
}




public String fractionToDecimal(int numerator, int denominator) {
	if (numerator == 0) {
		return "0";
	}
	if (denominator == 0) {
		return "";
	}
	HashSet<Integer> set = new HashSet<>();

	numerator = numerator / denominator;
	int left = numerator % denominator;
	StringBuilder sb = new StringBuilder();
	sb.append(String.valueOf(numerator));
	if (left == 0) {
		return sb.toString();
	}
	sb.append(".");

	String next = "";
	while (left != 0) {
		if (set.contains(left)) {
			int size = set.size();
			String result = sb.toString() + "(" + next + ")";
			return result;
		} else {
			set.add(left);
			while (left < denominator) {
				left *= 10;
			}
			next += String.valueOf(left / denominator);
			left %= denominator;
		}
	}
	return sb.toString() + next;

}



public static String parseCSV(String s) {

	int start = 0;
	int i = 0;
	List<String> result = new ArrayList<>();
	boolean quote = false;
	StringBuilder sb = new StringBuilder();
	for (i < s.length()) {
		char c = s.charAt(i);

		if (quote) {
			if (c == '"') {
				if (i == s.length() - 1) {
					result.add(sb.toString());
					return printStr(result);
				}
				if (i + 1 < s.length() && s.charAt(i + 1) == '"') {
					sb.append('"');
					i++;
				} else {
					i++;
					result.add(sb.toString());
					sb = new StringBuilder();
					quote = false;
				}
				
			} else {
				sb.append(c);
			}
		} else {
			if (c == ',') {
				result.add(sb.toString());
				sb = new StringBuilder();
			} else if (c == '"'){
				quote = true;
			} else {
				sb.append(c);
			}
		}
		i++;
	}

	if (sb.length() > 0) {
		result.add(sb.toString());
	}

	return printStr(result);
}





StringBuilder sb = new StringBuilder();
int i = 0;
while (i < str.length()) {
	char c = str.charAt(i);
	if (c == 'e') {
		sb.append(c);
		while (i + 1 < str.length() && str.charAt(i + 1)) {
			i++;
		}
		i--;
	} else {
		sb.append(c);
	}
	i++;
}
return sb.toString();


public List<String> topColor (String[][] input) {
	HashMap<String, Integer> map = new HashMap<>();

	int max = -1;
	for (int i = 0; i < input.length; i++) {
		for (int j = 0; j < input[0].length; j++) {
			map.put(input[i][j], map.containsKey(input[i][j]) + 1 : 1);
			max = Math.max(max, map.get(input[i][j]));
		}
	}
	List<String> result = new ArrayList<>();
	for (Map.Entry<String, Integer> entry : map.entrySet()) {
		if (entry.getValue() == max) {
			result.add(entry.getKey());
		}
	}
	Collections.sort(result, new Comparator<String>() {
		public int compare (String s1, String s2) {
			return s1 - s2;
		}
	});
	return result;
}



Map<String, List<Integer>> map = new HashMap<>();
int max = -1;
for (int i = 0; i < input.length; i++) {
	String str = input[i];
	if (!map.containsKey(str)) {
		List<String> list = new ArrayList<>();
		list.add(i);
		map.put(str, list);
	} else {
		map.put(str, map.get(str).add(str))
	}
	max = Math.max(map.get(str).size(), max);
}
List<String> result = new ArrayList<>();
for (List<Integer> list : map.values()) {
	if (list.size() == max) {
		for (Integer num : list) {
			result.add(input[num]);
		}
	}
}

return result;

public int helper (List<Integer> l1, List<Integer> l2) {
	if (l1.size() == 0) {
		return 0;
	}
	Map<Integer, Integer> map = new HashMap<>();
	for (int i = 0; i < l1.size(); i++) {
		int cur = l1.get(i);
		if (!map.containsKey(cur)) {
			map.put(cur, i);
		}
	}
	int max = Integer.MIN_VALUE;
	for (int i = 0; i < l2.size(); i++) {
		int cur = l2.get(i);
		if (map.containsKey(cur)) {
			max = Math.max(cur, i + map.get(cur));
		}
	}
	return max;
}


//preorder and postorder

int start = 0;
public TreeNode buildTree(int[] preorder, int[] inorder) {
	if () {

	}
	return helper (0, inorder.length - 1, preorder, inorder);
}
public TreeNode helper (int[] preorder, int[] inorder, int left, int right) {
	if (start > inorder.length || left > right) {
		return null;
	}

	TreeNode root = new TreeNode(preorder[start]);
	int pos = -1;
	for (int i = 0; i < inorder.length; i++) {
		if (inorder[i] == preorder[start]) {
			pos = i;
		}
	}
	start += 1;
	root.left = helper (preorder, inorder, left, i - 1); 
	root.right = helper (preorder, inorder, i + 1, right);
	return root;
}

public List<String> wordBreak(String s, Set<String> wordDict) {
	List<String>[] arrs = new ArrayList[s.length() + 1];
	arrs[0] = new ArrayList<>();
	
	for (int i = 0; i < s.length(); i++) {
		if (arrs[i] != null) {
			for (int j = i + 1; j <= s.length(); j++) {
				String cur = s.susbtring(i, j);
				if (wordDict.contains(cur)) {
					if (arrs[j] == null) {
						List<String> l = new ArrayList<>();
						l.add(cur);
						arrs[j] = l;
					} else {
						arrs[j].add(cur);
					}
				}
			}
		}
	}
	List<String> result = new ArrayList<>();
	if (arrs[s.length()] == null) {
		return result;
	} else {
		helper(arrs, result, s.length());
		return result;
	}

}
public void helper (List<String>[] arrs, List<String> result, int end) {

	if (end <= 0) {
		String path = result.get(result.size() - 1);
		for (i = result.size() - 2; i >= 0; i--) {
			path += " " + result.get(i);
		}
		result.add(path);
		return;
	}
	for (String word : arrs[end]) {
		result.add(word);
		helper(arrs, end - word.length(), result);
		result.remove(result.size() - 1);
	}

}

//Longest Absolute File Path
public int lengthLongestPath(String input) {
	String[] paths = input.split("\n");
	int[] stack = new int[paths.length + 1];
	int maxLen = 0;
	for (String s : paths) {
		int level = s.lastIndexOf("\t") + 1;
		int cur
	}
	return maxLen;
}


//mini parser
public NestedInteger deserialize(String s) {

	if (s.charAt(0) != '[') {
		return NestedInteger(Integer.parseInt(s));
	}
	Stack<NestedInteger> stack = new Stack<>();
	StringBuilder sb = new StringBuilder();
	while (pos < s.length()) {
		char c = s.charAt(pos);
		if (c == ',') {
			if (sb.length() != 0) {
				stack.peek().add(new NestedInteger(Integer.parseInt(sb.toString())));
				sb = new StringBuilder();
			}
		} else if (c == '[') {
			stack.push(new NestedInteger());
		} else if (c == ']') {
			if (sb.length() != 0 || sb == null) {
				stack.peek().add(new NestedInteger(Integer.parseInt(sb.toString())));
			}
			NestedInteger top = stack.pop();
			if (!stack.isEmpty()) {
				stack.peek().add(top);
			} else {
				return top;
			}
		} else {
			sb.append(c);
		}

	}
	if (sb.length() != 0 || sb == null) {
		return new NestedInteger(Integer.parseInt(sb.toString()));
	}
       return null;
}


//Frog Jump
public boolean canCross(int[] stones) {

	int k = 0;
	return helper(stones, 0, k);
}
public boolean helper (int[] stones, int index, int k) {

	if (index == stones.length - 1) {
		return true;
	}
	for (int i = k - 1; k <= k + 1; k++) {
		int nextJump = stones[index] + i;
		int nextPos = Arrays.binarySearch(stones, index + 1, stones.length, nextJump);
		if (nextPos > 0) {
			if (helper(stones, index + 1, i)) {
				return true;
			}
		}
	}
	return false;
}



//Lexicographical Numbers
public List<Integer> lexicalOrder(int n) {

	List<Integer> list = new ArrayList<>();
	for (int i = 1; i <= n; i++) {
		list.add(i);
		if (i * 10 <= n) {
			i *= 10;
		} else if (i + 1 <= n && i % 10 != 9) {
			i += 1;
		} else if (i % 10 != 9) {
			i = i / 10 + 1;
		} else {
			i += 1;
			while (i == (i / 10) * 10) {
				i /= 10;
			}
		}
	}
	return list;
}


//Longest Substring with At Least K Repeating Characters
//Longest Substring with At Least K Repeating Characters
public int longestSubstring(String s, int k) {
	if (s.length() < k) {
		return 0;
	}
	Map<Character, Integer> map = new HashMap<>();
	int count = 0;
	for (char c : s.toCharArray()) {
		map.put(c, map.containsKey(c) ? map.get(c) + 1 : 1);
		if (map.get(c) >= k) {
			count++;
		}
	}
	if (count == map.size()) {
    	return s.length();
    }

	Set<Character> splitSet = new HashSet<>();
	for (char c : map.keySet()) {
		if (map.get(c) < k) {
			splitSet.add(c);
		}
	}

    int start = 0;
    int i = 0;
    int max = 0;
    while (i < s.length()) {
    	char c = s.charAt(i);
    	if (splitSet.contains(c)) {
    		if (start != i) {
    			max = Math.max(max, longestSubstring(s.substring(start, i), k));
    		}
    		start = i + 1;
    	}
    	j++;
    }
    if (start != i) {
    	max = Math.max(max, longestSubstring(s.substring(start), k));
    }
    
    return max;
}


//Guess Number Higher or Lower II
public int getMoneyAmount(int n) {
        
	int[][] dp = new int[n + 1][n + 1];
	for (int len = 1; len < n; len++) {
		for (int i = 1; i + len <= n; i++) {
			int j = i + len;
			int min = Integer.MAX_VALUE;
			for (int k = i; k < j; k++) {
				int temp = k + Math.max(dp[i][k - 1], dp[k + 1][j]);
				min = Math.min(min, temp);
			}
			dp[i][j] = min;
		}
	}
	return dp[1][n];
}

3[a2[c]]  accaccacc

class strItem {
	int num = 0;
	StringBuilder sb = new StringBuilder();
	strItem (int num) {
		this.num = num;
	}
}

public String decodeString(String s) {
	int num = 0;
	Stack<strItem> stack = new Stack<>();
	stack.push(new strItem(1));
	for (char c : s.toCharArray()) {
		if (Character.isDigit(c)) {
			num = num * 10 c - '0';
		} else if (c == '[') {
			stack.push(new strItem(num));
			num = 0;
		} else if (c == ']') {
			String cur = stack.peek().sb.toString();
			int n = stack.pop().num;
			for (int i = 0; i < n; i++) {
				stack.peek().sb.append(cur);
			}
		} else {
			stack.peek().sb.append(c);
		}
	}
	return stack.pop().sb.toString();
}


//Rotate Function
Map<Integer, String> map = new HashMap<>();
public String numberToWords(int num) {

	fillMap();
	StringBuilder sb = new StringBuilder();

	if (num == 0) {
		return map.get(0);
	}

	if (num >= 1000000000) {
		int extra = num / 1000000000;
		sb.append(convert(extra)).append("Billion");
		num %= 1000000000;
	}

	if (num >= 1000000) {
		int extra = num / 1000000;
		sb.append(convert(extra)).append("Million");
		num %= 1000000;
	}
	if (num >= 1000) {
		int extra = num / 1000l
		sb.append(convert(extra)).append("Thousand");
		num %= 1000;
	}
	if (num > 0) {
		sb.append(convert(num));
	}
	return sb.toString().trim();
}

public String convert (int num) {
	StringBuilder sb = new StringBuilder();
	if (num >= 100) {
		int numHundred = num / 100;
		sb.append("" + map.get(numHundred)).append("Hundred");
		num %= 100;
	}
	if (num > 0) {
		if (num > 0 && num <= 20) {
			sb.append("" + map.get(num));
		} else {
			int numTen = num / 10;
			sb.append("" + map.get(numTen));
			if (num % 10 > 0) {
				sb.append("" + map.get(num % 10));
			}
		}
	}
	return sb.toString();
}
        
public void fillMap() {
		map.put(0, "Zero");
        map.put(1, "One");
        map.put(2, "Two");
        map.put(3, "Three");
        map.put(4, "Four");
        map.put(5, "Five");
        map.put(6, "Six");
        map.put(7, "Seven");
        map.put(8, "Eight");
        map.put(9, "Nine");
        map.put(10, "Ten");
        map.put(11, "Eleven");
        map.put(12, "Twelve");
        map.put(13, "Thirteen");
        map.put(14, "Fourteen");
        map.put(15, "Fifteen");
        map.put(16, "Sixteen");
        map.put(17, "Seventeen");
        map.put(18, "Eighteen");
        map.put(19, "Nineteen");
        map.put(20, "Twenty");
        map.put(30, "Thirty");
        map.put(40, "Forty");
        map.put(50, "Fifty");
        map.put(60, "Sixty");
        map.put(70, "Seventy");
        map.put(80, "Eighty");
        map.put(90, "Ninety");
}


// integer replacement
public int integerReplacement(int n) {

	return (int)helper(n);
        
}
public long helper (long n) {
	if (n < 3) {
		return n -1;
	} 
	if (n % 2 == 0) {
		return helper(n / 2) + 1;
	} else {
		return Math.min(helper(n - 1), helper(n + 1)) + 1;
	}

}

//binary watch
public List<String> readBinaryWatch(int num) {

	List<String> result = new ArrayList<>();
	for (int h = 0; h < 12; h++) {
		String hour = Integer.toBinaryString(h);
		for (int m = 0; m < 60; m++) {
			String mins = Integer.toBinaryString(m);
			String newStr = hour + mins;
			int count = 0;
			for (int i = 0; i < newStr.length(); i++) {
				if (newStr.charAt(i) == '1') {
					count++;
				}
			} 
			if (count == num) {
				String res = "";
				res += Stirng.valueOf(Integer.parseInt(hour, 2)); 
				res += ":"
				if (m < 0) {
					res += "0";
					res += Stirng.valueOf(Integer.parseInt(hour, 2));
				} else {
					res += Stirng.valueOf(Integer.parseInt(hour, 2));
				}
				result.add(res);
			}
		}
	}
	return result;
        
}


// is subsequence 


public boolean isSubsequence(String s, String t) {
        if (s.length() == 0 || s == null) {
            return true;
        }
        if (t.length() == 0 || t == null) {
            return false;
        }
        int i = 0;
        int j = 0;
        while (i < s.length() && j < t.length()) {
            char char_s = s.charAt(i);
            char char_t = t.charAt(j);
            if (char_s == char_t) {
                i++;
                j++;
            } else {
                j++;
            }
        }
        if (i < s.length()) {
            return false;
        }
        return true;
    }


//generalized abbreviation
public List<String> generateAbbreviations(String word) {

	List<String> result = new ArrayList<>();
	for (int i = 0; i < Math.pow(2, word.length()); i++) {
		String str = "";
		int num = i;
		int count = 0;
		for (int j = 0; j < word.length(); j++) {
			if ((num & 1) == 1) {
				count++;
				if (j == word.length() - 1) {
					str += String.valueOf(j);
				}
			} else {
				if (count != 0) {
					str += String.valueOf(count);
					count = 0;
				}
				str += word.charAt(j);
			}
			num >>= 1;
		}
		result.add(str);
	}
	return result;
}

//Android Unlock Patterns
int result = 0;
public int numberOfPatterns(int m, int n) {
	
	int[][] matrix=new int[10][10];
        matrix[1][3] = matrix[3][1] = 2;
        matrix[4][6] = matrix[6][4] = 5;
        matrix[7][9] = matrix[9][7] = 8;
        matrix[1][7] = matrix[7][1] = 4;
        matrix[2][8] = matrix[8][2] = 5;
        matrix[3][9] = matrix[9][3] = 6;
        matrix[1][9] = matrix[9][1] = 5;
        matrix[3][7] = matrix[7][3] = 5;
        boolean[] visited=new boolean[10];
        helper(visited,m,n,0,0,matrix);
        return result;
}
public void helper (boolean[] visited, int m, int n, int begin, int count, int[][] matrix) {

	if (count >= m) {
		result++;
	}
	if (count >= n) {
		return;
	}

	for (int i = 0; i <= 9; i++) {
		if (visited[i]) {
			continue;
		}
		int crossNum = matrix[begin][i];
		if (crossNum != 0 && !visited[crossNum]) {
			continue;
		}
		visited[i] = true;
		helper(visited, m, n, i, count + 1, matrix);
		visited[i] = false;
	}

}

//design hit counter
class TimeSlot {
    private long time;
    private long hits;
    public TimeSlot(long t) {
        time = t;
        hits = 1;            
    }      
}

public class HitCounter {
    
    int count;
    TreeMap<Integer, Integer> map;
    HitCounter() {
        this.count = 0;
        this.map = new TreeMap<>();
    }
   
    public void hit(int timestamp) {
        if (map.containsKey(timestamp)) {
        	map.put(timestamp, map.get(timestamp) + 1);
        	count++;
        	return;
        } else {
        	check(timestamp);
        	map.put(timestamp, 1);
        	count++;
        	return;
        }

    }

    public long getHits(int timestamp) {
        check(timestamp);
        return count;
    }

    public void check (int timestamp) {
    	while (!map.size() == 0 && timestamp - map.firstKey() >= 300) {
    		count -= map.get(map.firstKey());
    		map.remove(map.firstKey());
    	}
    }
}





//Boom Enemy

public int maxKilledEnemies (char[][] grid) {
    if (grid == null || grid.length == 0 || grid[0].length == 0) {
        return 0;
    }
    int result = 0;
    int row = grid.length;
    int col = grid[0].length;
    int rowCnt = 0;
    int[] colCnt = new int[col];

    for (int i = 0; i < row; i++) {
    	for (int j = 0; j < col; j++) {
    		if (j == 0 || grid[i][j - 1] == 'W') {
    			rowCnt = 0;
    			for (int k = j; k < col && grid[i][k] == 'W'; k++) {
    				if (grid[i][k] == 'E') {
    					rowCnt += 1;
    				}
    			}
    		}
    		if (i == 0 || grid[i - 1][j] == 'W') {
    			colCnt[j] = 0;
    			for (int k = i; k < row && grid[k][j] == 'W'; k++) {
    				if (grid[k][j] == 'E') {
    					colCnt[j] += 1;
    				}
    			}
    		}
    		if (grid[i][j] == '0') {
    			result = Math.max(result, rowCnt + rowCnt[j]);
    		}
    	}
    }
    return result;
}
 public int maxKilledEnemies (char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        int ret = 0;
        int row = grid.length;
        int col = grid[0].length;
        int rowCache = 0;
        int[] colCache = new int[col];
        
        for (int i = 0;i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (j == 0 || grid[i][j-1] == 'W') {
                    rowCache = 0;
                    for (int k = j; k < col && grid[i][k] != 'W'; k++) {
                        rowCache += grid[i][k] == 'E' ? 1 : 0;
                    }
                }
                if (i == 0 || grid[i-1][j] == 'W') {
                    colCache[j] = 0;
                    for (int k = i;k < row && grid[k][j] != 'W'; k++) {
                        colCache[j] += grid[k][j] == 'E' ? 1 :0;
                    }
                }
                if (grid[i][j] == '0') {
                    ret = Math.max(ret, rowCache + colCache[j]);
                }
            }
        }
        return ret;
    }


//sort transfomed array
public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
	if (nums.length == 0 || nums == null) {
		return nums;
	}
	int[] result = new int[nums.length];
	
	int start = 0;
	int end = nums.length - 1;
	int nextIndex = 0;

	if (a > 0 || (a == 0 && b >= 0)) {
		nextIndex = end;
	} else if (a < 0 || (a == 0 && b <= 0)) {
		nextIndex = start;
	}

	while (start <= end) {
		if (Math.abs(nums[start] - mid) > Math.abs(nums[end] - mid)) {
			nums[nextIndex--] = nums[start];
		} else {

		}
		if (a > 0 || (a == 0 && b >= 0)) {
			if (Math.abs(nums[start] - mid) > Math.abs(nums[end] - mid)) {
				int x = nums[start];
				result[nextIndex--] = a * x * x + b * x + c;
			} else {
				int x = nums[end];
				result[nextIndex--] = a * x * x + b * x + c;
			}
		} else if (a < 0 || (a == 0 && b <= 0)) {
			if (Math.abs(nums[start] - mid) > Math.abs(nums[end] - mid)) {
				int x = nums[start];
				result[nextIndex++] = a * x * x + b * x + c;
			} else {
				int x = nums[end];
				result[nextIndex++] = a * x * x + b * x + c;
			}
		}
	}
	
	return result;

}


//Find Leaves of Binary Tree (Java)
int count = 0;
public List<List<Integer>> findLeaves(TreeNode root) {

	List<List<Integer>> result = new ArrayList<>();
	helper(root, result);
	return result;
}
public int helper (TreeNode root, List<List<Integer>> result) {
	if (root == null) {
		return -1;
	}

	int left = helper(root.left, result);
	int right = helper(root.right, result);

	int max = Math.max(left, right) + 1;

	if (result.size() <= max) {
		result.add(new ArrayList<>());
	}
	result.get(max).add(root.val);
	return max;
}

//plus one linked list
public ListNode plusOne(ListNode head) {
	if (head == null) {
		return null;
	}
	ListNode newHead = reverse(head);
	ListNode node = newHead;
	int carry = 1;
//////////
	while (newHead != null) {
		if (newHead.val + 1 <= 9) {
			newHead.val += 1;
			break;
		} else {
			newHead.val = 0;
			if (newHead.next == null) {
				newHead.next = new ListNode(1);
				break;
			}
			newHead = newHead.next;
		}
	}
	return reverse(node);
///////////
	while (newHead != null) {
		int sum = newHead.val + carry;
		carry = sum / 10;
		if (carr == 0) {
			newHead.val = sum;
			return reverse(node);
		} else {
			newHead.val = sum % 10;
		}
		newHead = newHead.next;
	}
	ListNode dummy = new ListNode(1);
	dummy.next = reverse(node);
	return dummy;
}
public ListNode (ListNode head) {
	if (head == null || head.next == null) {
		return head;
	}
	ListNode tail = null;
	while (head != null) {
		ListNode next = head.next;
		head.next = tail;
		tail = head;
		head = next;
	}
	return tail;
}

//Range Addition

public int[] getModifiedArray(int length, int[][] updates) {
	int[] temp = new int[length];
	for (int[] update : updates) {
		int start = update[0];
		int end = update[1];
		int inc = update[2];
		temp[start] += inc;
		if (end + 1 < length) {
			temp[end + 1] += inc * (-1);	
		} 
	}
	for (int i = 1; i < length; i++) {
		temp[i] += temp[i - 1];
	}
	return temp;
}



//Design Phone Directory
public class PhoneDirectory {
	int max;
	// used one
	Set<Integer> set;
	// unused ones
	LinkedList<Integer> queue;
	public PhoneDirectory(int maxNumbers) {
		this.queue = new LinkedList<.();
		for (int i = 0; i < maxNumbers; i++) {
			queue.offer(i);
		}
		this.set = new HashSet<>();
		this.max = maxNumbers - 1;
	}
	public int get() {
		if (queue.isEmpty()) {
			return -1;
		}
		int e = queue.poll();
		set.add(e);
		return e;
	}
	public boolean check(int number) {
		return !set.contains(number) && number <= max;
	}
	public void release (int number) {
		if (set.contains(number)) {
			set.remove(number);
			queue.offer(number);
		}
	}
}
// rearrange string k distance apart
public String rearrangeString(String str, int k) {

	Map<Character, Integer> map = new HashMap<>();
	for (int i = 0; i < str.length(); i++) {
		char c = str.charAt(i);
		map.put(c, map.containsKey(c) ? map.get(c) + 1 : 1);
	}
	PriorityQueue<Character> max = new PriorityQueue<Character>(new Comparator<Character>() {
		public int compare (Character c1, Character c2) {
			return map.get(c1).intValue() - map.get(c2).intValue();
		}
	});
	for (char c : map.keySet()) {
		queue.offer(c);
	}
	StringBuilder sb = new StringBuilder();

	int len = str.length();

	while (!deque.isEmpty()) {
		int cnt = Math.min(k, len);
		List<Character> temp = new ArrayList<>();
		
		for (int i = 0; i < cnt; i++) {
			if (queue.isEmpty()) {
				return "";
			}
			char c = queue.poll();
			sb.append(c);
			map.put(c, map.get(c) - 1);
			if (map.get(c) > 0) {
				temp.add(c);
			}
			len--;
		}
		for (char c : temp) {
			queue.offer(c);
		}

	}
	return sb.toString();
}

//Line reflection
public boolean isReflected(int[][] points) {
	if (points.length == 0 || points == null) {
		return true;
	}
	int max = Integer.MIN_VALUE;
	int min = Integer.MAX_VALUE;
	Set<String> set = new HashSet<>();
	for (int [] point : points) {
		set.add(point[0] + "," + point[1]);
		min = Math.min(min, point[0]);
		max = Math.max(max, point[0]);
	}
	int sum = min + max;
	for (int[] point : points) {
		if (!set.contains(sum - point[0] + "," + point[1])) {
			return false;
		}
	}
	return true;
}

//Number of Islands II
public List<Integer> numIslands2(int m, int n, int[][] positions) {

	int[] rootArray = new int[m * n];
	Arrays.fill(rootArray, -1);

	List<Integer> result = new ArrayList<>();
	int[][] direction = new int[][]{{-1,0}, {1,0},{0,1},{0,-1}};
	int count = 0;
	for (int k = 0; k < positions.length; k++) {
		count++;
		int[] p = positions[k];
		int index = p[0] * n + p[1];
		rootArray[index] = index;

		for (int r = 0; r < 4; r++) {
			int x = p[0] + direction[r][0];
			int y = p[1] + direction[r][1];

			if (0 <= x && x < m && 0 <= y && y < n && rootArray[x * n + y] != -1) {
				int thisRoot = getRoot(rootArray, x * n + y);
				if (thisRoot != index) {
					rootArray[thisRoot] = index;
					count--;
				}
			}
		}
		result.add(count);
	}
	return result;
}
public int getRoot (int[] arr, int i) {
	while (i != arr[i]) {
		i = arr[i];
	}
	return i;
}

// numbers of lakes
public int numofLakes(int[][] grid) {
	int[][] temp = grid;
	int count = 1;
	for () {
		for () {
			if ( == 1) {
				helper(count, i, j, temp);
				count++;
			}
		}
	}

	count = 0;
	for () {
		for () {
			if ( == 0) {
				int curId = 0;
				if (i > 0) {
					curId = (temp[i - 1][j] != 0 ? world[i - 1][j] : curId);
				}



				if (findLake(temp, i, j, curId)) {
					count++;
				}
			}
		}
	}
	return count;
}

public int findLake () {
	temp[x][y] = curId;
	
	boolean up = i != 0 && (temp[x - 1][y] == curId) || (temp[x - 1][y] == 0 && findLake(temp, i - 1, j , curId))

	int left = 
	int right = 
	int top = 
	int down = 
}


public void helper() {
	
	if (x < 0 || x >= temp.length || y < 0 || y >= temp[0].length) {
		return;
	}
	if (temp[x][y] != 1) {
		return;
	}
	temp[x][y] = count;
	helper();
	helpr();
}

//Graph Valid Tree
public boolean validTree(int n, int[][] edges) {
	if (edges.length == 0 || edges[0].length == 0 || edges == null) {
		return true;
	}

	List<List<Integer>> list = new ArrayList<>();
	for (int i = 0; i < n; i++) {
		list.add(new ArrayList<>());
	}
	for (int[] edge : edges) {
		list.get(edge[0]).add(edge[1]);
		list.get(edge[1]).add(edge[0]);
	}

	boolean[] visited = new boolean[n];

	Deque<Integer> deque = new LinkedList<>();
	deque.offer(0);
	while (!deque.isEmpty()) {
		int cur = deque.poll();
		if (visited[cur]) {
			return false;
		}
		visited[cur] = true;
		List<Integer> l = list.get(cur);
		for (int num : l) {
			if (!visited[num]) {
				deque.offer(num);
			}
		}

	}

	for (int i = 0; i < n; i++) {
		if (!visited[i]) {
			return false;
		}
	}

	return true;
}


//Design Snake Game (Java)
public class SnakeGame {
	Deque<Integer> deque;
	Set<Integer> body;
	int score;
	int x;
	int y;
	int row;
	int col;
	int[][] food;
	int index;
	public SnakeGame(int width, int height, int[][] food) {
		this.body = new HashSet<>();
		body.add(0);
		this.index = 0;
		this.score = 0;
		this.food = food;
		this.x = 0;
		this.y = 0;
		this.row = width;
		this.col = height; 
		this.deque = new LinkedList<>();
		deque.offer(0);
	}
	public int move(String direction) {
		int h = deque.getFirst() / width;
		int w = deque.getFirst() % width;

		switch (direction) {
			case "U" :
				h--;
				break;
			case "D" :
				h++;
				break;
			case "L" :
				w--;
				break;
			case "R" :
				w++;
				break;
		}

		if (h < 0 || h >= row || w < 0 || w >= col) {
			return -1;
		}
		int head = h * width + w;
		deque.addFirst(head);
		if (index < food.length && h == food[index][0] && w == food[index][1]) {
			score++;
			index++;
		} else {
			body.remove(queue.removeLast());
		}
		if (!body.add(head)) {
			return -1;
		}
		return score;
	}
}


//Flatten 2D Vector
public class Vector2D {
	List<Iterator<Integer>> it;
	int cur = 0;
	public Vector2D(List<List<Integer>> vec2d) {
		this.it = new ArrayList<>();
		for (List<Integer> list: vec2d) {
			if (list.size() > 0) {
				it.add(list.iterator());
			}
			
		}
	}
	public int next() {
		Integer res = its.get(cur).next();
		if (!its.get(cur).hasNext()) {
			cur++;
		}
		return res.intValue();
	}
	public boolean hasNext() {
		return cur < its.size() && its.get(cur).hasNext();
	}
}


//Find the Celebrity
public class Solution extends Relation {
	public int findCelebrity(int n) {
		if (n <= 1) {
			return -1;
		}
		int left = 0;
		int right = n - 1;
		while (left < right) {
			if (knows(left,right)) {
				left++;
			} else {
				right--;
			}
		}
		int candidate = right;
		for (int i = 0; i < n; i++) {
			if (i != candidate && (!knows(i, candidate) || knows(candidate, i)) {
				return -1;
			}
		}
		return candidate;
	}
}

//Design Tic-Tac-Toe
public class TicTacToe {
	int n;
	int[] rows;
	int[] cols;
	int diag;
	int xdiag;
	
	public TicTacToe (int n) {
		this.n = n;
		this.rows = new int[n];
		this.cols = new int[n];
		this.diag = 0;
		this.xdiag = 0;
	}
	public int move(int row, int col, int player) {
		int count = player == 1 ? 1 : -1;
		rows[row] += count;
		col[col] += count;
		if (row == col) {
			diag += count;
		}	
		if (row + col == n - 1) {
			xdiag += count;
		}
		if (Math.abs(rows[row]) == n || Math.abs(cols[col]) == n || Math.abs(diag) == n|| Math.abs(xdiag) == n) {
			return count > 0 ? 1 : 2;
		}
		return 0;
	}
	
}


//Count Univalue Subtrees
public class Solution {
	int count = 0;  
	public int countUnivalSubtrees(TreeNode root) { 

		helper(root);
		return count;
	}
	public boolean helper (TreeNode root) {
		if (root == null) {
			return true;
		}
		if (root.left == null && root.right == null) {
			count++;
			return true;
		}
		boolean left = helepr(root.left);
		boolean right = helper(root.right);
		
		if (left && right && (root.left == null || root.left.val == root.val) && (root.right == null || root.right.val == root.val)) {
			count++;
			return true;
		}
		return false;
	}
}


// count of tange sum

public int countRangeSum(int[] nums, int lower, int upper) {

	int n = nums.length;
	long[] sums = new long[n + 1];
	for (int i = 0; i < nums.length; i++) {
		sums[i + 1] = sum[i] + nums[i];
	}
	return helper (sums, 0, n + 1, lower, upper);
}
public int helper (long[] sums, int start, int end, int lower, int upper) {
	if (end - start <= 1) {
		return 0;
	}
	int mid = start + (end - start) / 2;  
	int count = helper (sums, start, mid, lower, upper) + helper(sums, mid, end, lower, upper)
	int j = mid, k = mid, t = mid;
	long[] cache = new long[end - start];


}


//Closest Binary Search Tree Value
public int closestValue(TreeNode root, double target) {
	if (root == null) {
		return 0;
	}
	int result = root.val;
	return Math.abs(helper(root, target) - target) < Math.abs(root.val - target) ? helper(root, target) : root.val;
}
public int helper (TreeNode root, double target) {
	if (root.left == null && root.right == null) {
		return root.val;
	}
	int result = -1;
	if (taget < root.val) {
		result = helper(root.left, target);
	} else if (target > root.val) {
		result = helper(root.right, target);
	} else {
		result = root.val;
	}
	return result;
}

//Closest Binary Search Tree Value II
PriorityQueue<Integer> max;
PriorityQueue<Integer> min;
public List<Integer> closestKValues(TreeNode root, double target, int k) {

	this.max = new PriorityQueue<>(new Comparator<Integer>() {
		public int compare (Integer i1, Integer i2) {
			return i2 - i1;
		}
	})
	this.min = new PriorityQueue<>();

	getSuccessor(root, target, k, max);
	getPredecessor(root, target, k, min);
	List<Integer> result = new ArrayList<>();
	int i = 0;
	while (i < k) {
		if (max.size() == 0 {
			result.add(min.poll());
		} else if (min.size() == 0) {
			result.add(max.poll());
		} else {
			int min = min.peek();
			int max = max.peek();
			if (Math.abs(min - target) < Math.abs(max - target)) {
				result.add(min.poll());
			} else{
				result.add(max.poll());
			}
		}
		i++;
	}
	return result;
}
public void getPredecessor (TreeNode root, double target, int k, PriorityQueue<Integer> max) {
	if (root == null) {
		return;
	}
	getPredecessor(root.left, target, k, max);
	if (root.val > target) {
		return;
	}
	max.offer(root.val);
	getPredecessor(root.right, target, k, max);
}

public void getSuccessor (TreeNode root, double target, int k, PriorityQueue<Integer> min) {
	if (root == null) {
		return;
	}
	getSuccessor(root.right, target, k, min);
	if (root.val < target) {
		return;
	}
	min.offer(root.val);
	getSuccessor(root.left, target, k, min);
}

//Leetcode: Binary Tree Upside Down
public TreeNode UpsideDownBinaryTree(TreeNode root) {
	if (root == null) {
		return null;
	}
	return helper (root, null);
}
public TreeNode helper (TreeNode root, TreeNode parent) {
	if (root == null) {
		return parent;
	}
	TreeNode newRoot = helper(root.left, root);
	root.left = parent == null ? null : parent.right;
	root.right = parent;

	return newRoot;
}


//Nested List Weight Sum
public int depthSum(List<NestedInteger> nestedList) {
	return helper (netedList, 1);
}	
public int helper (List<NestedInteger> nestedList, int depth) {
	if (nestedList.size() == 0 || nestedList == null) {
		return 0;
	}
	int sum = 0;
	for (NestedInteger n : nestedList) {
		if (n.isInteger()) {
			sum = n.getInteger() * depth;
		} else {
			sum = helepr(n, depth + 1);
		}
	}
	
	return sum;
}


//Nested List Weight Sum II
Map<Integer, List<Integer>> map = new HashMap<>();
int max = Integer.MIN_VALUE;
public int depthSumInverse(List<NestedInteger> nestedList) {
	helper (map, nestedList, 1);
	int result = 0;
	for (int i = max; i >= 1; i--) {
		if (map.get(i) != null) {
			for (int v : map.get(i)) {
				result += v * (max - i + 1);
			}
		}
	}

	return result;
}
public void helper (NestedInteger n, int depth) {
	if (n.size() == 0 || n == null) {
		return;
	}
	max = Math.max(depth, max);
	for (NestedInteger n : nestedList) {
		if (n.isInteger()) {
			if (map.containsKey(depth)) {
				List<Integer> l = map.get(depth);
				l.add(n.getInteger);
				map.put(depth, l);
			} else {
				List<Integer> l = new ArrayList<>();
				l.add(n.getInteger());
				map.put(depth, l);
			}
		} else {
			helepr(n, depth + 1);
		}
	}
}
//Palindrome Permutation II
public List<String> generatePalindromes(String s) { 
	List<String> result = new ArrayList<>();
	char[] arr = new char[256];
	int min = Integer.MAX_VALUE;
	int max = Integer.MIN_VALUE;
	for (int i = 0; i < s.length(); i++) {
		arr[i] += 1;
		min = Math.min(min, arr[i]);
		max = Math.max(max, arr[i]);
	}
	int odd = 0;
	int oddPos = -1;
	for (int i = min; i <= max; i++) {
		if (odd == 0 && arr[i] % 2 == 1) {
			odd++;
			oddPos = i;
		} else if (arr[i] % 2 == 1) {
			return result;
		}
	}
	
	String str = "";
	if (oddPos != -1) {
		str += (char) oddPos;
		arr[oddPos]--;
	}
	helper(arr, min, max, str, result, s.length());
	return result;
}
public void helper (char[] arr, int min, int max, String str, List<String> result, int len) {
	if (str.length() == len) {
		result.add(new String(str));
		return;
	}
	for (int i = min; i <= max; i++) {
		if (arr[i] > 0) {
			arr[i] -= 2;
			str = (char)i + str + (char)i;
			helper (arr, min, max, str, result, len);
			str = str.substring(1, str.length() - 1);
			arr[i] += 2;

		}
	}
}


//Palindrome Permutation
public boolean canPermutePalindrome(String s) {
	Set<Character> set = new HashSet<>();
	for (int i = 0; i < s.length(); i++) {
		char c = s.cahrAt(i);
		if (!set.contains(c)) {
			set.add(c);
		} else {
			set.remove(c);
		}
	}
	return set.size() <= 1;
}


//Range Sum Query 2D - Mutable
public class NumMatrix {
	int[][] sums;
	public NumMatrix(int[][] matrix) {
		this.sums = new int[matrix.length][matrix[0].length];
		for (int i = 0; i < matrix.length; i++) {
			int sum = 0;
			for (int j = 0; j < matrix[0].length; j++) {
				sum += matrix[i][j];
				sums[i][j] = sum;
			}
		}
	}
	public void update(int row, int col, int val) {
		int diff = matrix[row][col] - val;
		for (int j = col; j < matrix[0].length; j++) {
			sums[row][j] += diff;
		}
		
	}	
	public int sumRegion(int row1, int col1, int row2, int col2) {
		int result = 0;
		for (int i = row1; i <= row2; i++) {
			result += col1 == 0 ? sums[i][col2] : sums[i][col2] - sums[i][col1 - 1];
		}
		return result;
	}
}


//Read N Characters Given Read4 I
public int read(char[] buf, int n) {
	for (int i = 0; i < n; i += 4) {
		char[] temp = new char[4];
		int size = read4(temp);
		System.arraycopy(temp, 0, buf, 0, Math.min(n - i, size));
		if (size < 4) {
			return Math.min(n, i + size);
		}
	}
	return n;
}

//Read N Characters Given Read4 II - Call multiple times

public class Solution extends Reader4 {
	Queue<Character> remain = new LinkedList<>();
	public int read(char[] buf, int n) {
		int i = 0;
		while (i < n && !remain.isEmpty()) {
			buf[i] = remain.poll();
			i++;
		}
		for (; i < n; i += 4) {
			char[] temp = new char[4];
			int size = read4(temp);
			if (size + i > n) {
				System.arraycopy(temp, 0, buf, i, n - i);
				for (int j = n - i; j < size; j++) {
					remain.offer(temp[j]);
				}
			} else {
				System.arraycopy(temp, 0, buf, i, size);
			}
			if (size < 4) {
				return Math.min(n, i + size);
			}
		}
		return n;
	}

}


//Longest Substring with At Most Two Distinct Character
public int lengthOfLongestSubstringTwoDistinct(String s) {  
	Map<Character, List<Integer>> map = new HashMap<>();
	int start = 0;
	int i = 0;
	int result = Integer.MIN_VALUE;
	for (i < s.length()) {
		char c = s.charAt(i);
		if (map.size() > 2) {

			result = Math.max(result, i - start);

			while (map.size() > 2) {
				List<Integer> temp = map.get(s.charAt(start));
				List<Integer> copy = new ArrayList<>();
				for (Integer i : temp) {
					if (i != start) {
						copy.add(i);
					}
				}
				if (copy.size() == 0) {
					map.remove(s.charAt(start));
				} else {
					map.put(c, copy);
				}
				start++;
			}

		} else {
			if (!map.containsKey(c)) {
				List<Integer> temp = new ArrayList<>();
				temp.add(i);
				map.put(c, temp);
			} else {
				List<Integer> list = map.get(c);
				list.add(i);
				map.put(c, list);
			}	
		}
		i++;
	}
	return result == Integer.MIN_VALUE ? 0 : result;

}


//Maximum Size Subarray Sum Equals k
public int maxSubArrayLen(int[] nums, int k) {
	Map<Integer, Integer> map = new HashMap<>();
	map.put(0, -1);
	int sum = 0;
	for (int i = 0; i < nums.length; i++) {
		sum += nums[i];
		if (!map.containsKey(sum)) {
			map.put(sum, i);
		}
		if (map.containsKey(sum - k)) {
			result = Math.max(result, i - map.get(sum - k));
		}
	}
	int result == Integer.MIN_VALUE ? 0 : result;
}

//Number of Connected Components in an Undirected Graph

public int countComponents(int n, int[][] edges) {
	int count = n;
	int[] root = new int[n];
	for (int i = 0; i < n; i++) {
		root[i] = i;
	}
	for (int i = 0; i < edges.length; i++) {
		int x = edges[i][0];
		int y = edges[i][1];

		int xRoot = getRoot(root, x);
		int yRoot = getRoot(root, y);

		if (xRoot != yRoot) {
			count--;
			root[xRoot] = yRoot;
		}

	}
	return count;

}
public int getRoot (in[] arr, int i) {
	while (arr[i] != i) {
		arr[i] = arr[arr[i]];
		i = arr[i];
	}
	return i;
}

// increasing triplet subsequence
public boolean increasingTriplet(int[] nums) {
	int left = Integer.MAX_VALUE;
	int mid = Integer.MAX_VALUE;
	for (int num : nums) {
		if (num <= left) {
			left = min;
		} else if (num <= mid) {
			mid = num;
		} else {
			return true;
		}
	}
	return false;
}
// inorder successor
public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {

	// 
	if (p.right != null) {
		TreeNode cur = p.right;
		while (cur != null) {
			cur = cur.left;
		}
		return cur;
	} else {
		Stack<TreeNode> stack = new Stack<>();
		TreeNode node = root;
		while (node != p) {
			stack.push(node);
			node = node.left;
		}

		while (!stack.isEmpty()) {
			TreeNode cur = stack.pop();
			if (cur == p) {
				return stack.peek();
			}
			if (cur.right != null) {
				TreeNode temp = cur;
				if (temp.right == p) {
					return stack.pop();
				}
				while (temp != null) {
					stack.push(temp);
					temp = temp.left;
				}
				
			}
		}

	}

	return null;
}

// largetest BST subtree
public class Solution {
	class Node {
		int size;
		int min;
		int max;
		boolean isBST;
		Node () {
			this.min = Integer.MAX_VALUE;
			this.max = Integer.MIN_VALUE;
			this.isBST = false;
			this.size = 0;
		}
	}
	public int largestBSTSubtree(TreeNode root) {
		return helepr(root).size;
	}
	
	public Node helper (TreeNode root) {
		Node cur = new Node();
		if (root == null) {
			cur.isBST = true;
			return cur;
		}
		
		Node left = helepr(root.left);
		Node right = helper(root.right);
		
		cur.min = Math.min(root.val, left.min);
		cur.max = Math.max(root.val, right.max);

		if (left.isBST && right.isBST && left.max <= root.val && root.val <= right.min) {
			cur.size = left.size + right.size + 1;
			cur.isBST = true;
		} else {
			cur.size = Math.max(left.size, right.size);
			cur.isBST = false;
		}
		return cur;
	}
}



// group shifted string 
public List<List<String>> groupStrings(String[] strings) {
	List<List<String>> result = new ArrayList<>();
	Map<String, List<String>> map = new HashMap<>();
	for (String str : strings) {
		StringBuilder sb = new StringBuilder();
		sb.append('0');
		for (int i = 1; i < str.length(); i++) {
			int pre = str.charAt(i - 1) - 'a';
			int cur = str.charAt(i) - 'a';
			if (cur >= pre) {
				sb.append(cur - pre);
			} else {
				sb.append(cur- pre + 26);
			}
		}
		if (map.containsKey(sb.toString())) {
			List<String> list = map.get(sb.toString());
			list.add(str);
			map.put(sb.toString(), list);
		} else {
			List<String> list = new ArrayList<>();
			list.add(str);
			map.put(sb.toString(), list);
		}
	}


	for (List<String> list: map.values()) {
		result.add(new ArrayList<>(list));
	}
	return result;

}



//One Edit Distance
public boolean isOneEditDistance(String s, String t) {
	if (Math.abs(s.length() == t.length())) {
		int count = 0;
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) != t.charAt(i)) {
				count++;
			}
		}
		return count == 1;
	} else if (s.length() - t.length() == 1) {
		return helper(s, t);
	} else if (s.length() - t.length() == 1) {
		return helper(t, s);
	} 
	return false;
}
public boolean helepr (String l, String s) {
	boolean result = false;
	for (int i = 0; i < l.length); i++ {
		if (l.charAt(i) != s.charAt(i)) {
			return l.susbtring(l + 1).equals(s.susbtring(l));
		}
	}
	return true;
}

//Paint Fence
public int numWays(int n, int k) {

	int[] dp = {0, k, k* k, 0};
	if (n <= 2) {
		return dp[n];
	}
	for (int i = 2; i < n; i++) {
		dp[3] = (k - 1) * (dp[1] + dp[2]);
		dp[1] = dp[2];
		dp[2] = dp[3];
	}
	return dp[3];
}


//paint house I
public int minCost(int[][] costs) {
	if (costs.length == 0 || costs[0].length == 0 || costs == null) {
		return 0;
	}
	for (int i = 1; i < costs.length; i++) {
		costs[i][0] += Math.min(costs[i][1], costs[i][2]);
		costs[i][1] += Math.min();
		costs[i][2] += Math.min();
	}
	return Math.min(Math.min(costs[costs.length - 1][0], costs[costs.length - 1][1]), costs[costs.length - 1][2]);
}


//paint house II
public int minCostII (int[][] costs) {
	int preMin = 0;
	int preSecMin = 0;
	int preIndex = -1;
	for (int i = 0; i < costs.length; i++) {
		int curMin = Integer.MAX_VALUE;
		int curSecMin = Integer.MAX_VALUE;
		int curIndex = -1;
		for (int j = 0; j < costs[0].length; j++) {
			costs[i][j] += (preIndex == j ? preSecMin : preMin);
			if (costs[i][j] < curMin) {
				curMin = costs[i][j];
				curSecMin = curMin;
				curIndex = j;
			} else if (costs[i][j] < curSecMin) {
				curSecMin = costs[i][j];
			}
		}
		preMin = curMin;
		preSecMin = curSecMin;
		preIndex = curIndex;
	}
	return preMin;

}

//Metting rooms I
public boolean canAttendMeetings(Interval[] intervals) {
	if(intervals == null || intervals.length == 0) {
		return true;
	}
	Arrays.sort(new Comparator<Interval i>() {
		public int compare (Interval i1, Interval i2) {
			return i1.start - i2.start;
		}
	});
	int end = intervals[0].end;
	for (int i = 1; i < intervals.length; i++) {
		if (interval[i].start < end) {
			return false;
		}
		end = Math.max(end, intervals[i].end);
	}
	return true;
}

// Meeting room II
public int minMeetingRooms(Interval[] intervals) {

}

//Shortest Word Distance
public int shortestDistance(String[] words, String word1, String word2) {
	int left = -1;
	int right = -1;
	int result = Integer.MAX_VALUE;
	for (int i = 0; i < words.length; i++) {
		String str = words[i];
		if (str.equals(word1)) {
			left = i;
			if (right != -1) {
				result = Math.max(Math.abs(left - right), result);
			}
		} else if (str.equals(word2)) {
			right = i;
			if (lef != -1) {
				result = Math.max(result, Math.abs(left - right));
			}
		}
	}
	return result;
}
//Shortest Word Distance II
public class WordDistance {
	public WordDistance(String[] words) {

	}
	public int shortest(String word1, String word2) {

	}
}
public int shortestWordDistance(String[] words, String word1, String word2) {

}

// Strobogrammatic Number II
public List<String> findStrobogrammatic(int n) {
	List<String> result = new ArrayList<>();
	if (n == 0) {
		return result;
	}
	if (n == 1) {
		return new ArrayList<String>(Arrays.asList("0", "1", "8"));
	}
	return helper(n, n);

}
public List<String> helper (int n, int m) {
	if (n == 0) {
		return new ArrayList<>(Arrays.asList(""));
	}
	if (n == 1) {
		return new ArrayList<String>(Arrays.asList("0", "1", "8"));
	}
	List<String> list = helper(n - 2, m);
	List<String> result = new ArrayList<>();
	for (int i = 0; i < list.size(); i++) {
		String s = list.get(i);
		if (n != m) {
			result.add("0" + s + "0");
			result.add("1" + s + "1");
            result.add("8" + s + "8");
            result.add("6" + s + "9");
            result.add("9" + s + "6");
		}
	}
	return result;
}

//Strobogrammatic Number II
public class Solution {
    private List<String> result = new ArrayList<String>();
    private Map<Character, Character> map = new HashMap<>();

    public List<String> findStrobogrammatic(int n) {
    	fillMap(map);

    	char[] arr = new char[n];
    	helper(arr, 0, n - 1);
    	return result;
    }
    public void helper (char[] arr, int start, int end) {
    	if (start > end) {
    		if (arr.length == 1 || (arr.length > 1 && arr[0] != '0')) {
    			result.add(new String(arr));
    		}
    		return;
    	}
    	for (Character c : map.keySet()) {
    		arr[start] = c;
    		arr[end] = map.get(c);
    		if (start < end || (start == end && map.get(c) == c)) {
    			helper (arr, start + 1, end - 1);
    		}
    	}

    }
    public void fillMap(Map<Character, Character> map) {
    	map.put('0', '0');
        map.put('1', '1');
        map.put('8', '8');
        map.put('6', '9');
        map.put('9', '6');
    }

}

//Strobogrammatic Number III
int count = 0;
Map<Character, Character> map = new HashMap<>();
public int strobogrammaticInRange(String low, String high) {

	if (low == null || low.length() == 0 || high == null || hight.length() == 0) {
		return 0;
	}
	fillMap();
	for (int i = low.length(); i < hight.length(); i++) {
		char[] arr = new char[n];
		helper(arr, 0, i - 1, low ,high);
	}
	return count;
}
public void helepr (char[] arr, int start, int end, String low, String high) {
	if (start > end) {
		String str = new String(arr);
		if (arr.length == 1 || (arr.length > 1 && arr[0] != '0') && comapre(low, str) && compare(str, high)) {
			count++;
		}
	}	
	for (Character c : map.keySet()) {
		arr[start] = c;
		arr[end] = map.get(c);
		if ((start == end && s == e) || start < end) {
			helper(arr, start + 1, end - 1, low, high);
		}
	}
}
public boolean compare (String low, String high) {
	if (s1.length() == s2.length()) {
		if (s1.compareTo(s2) <= 0) {
			return true;
		} else {
			return false;
		}
	}
	return true;
}

public void fillMap() {
	map.put('0', '0');
    map.put('1', '1');
    map.put('8', '8');
    map.put('6', '9');
    map.put('9', '6');
}

// walls and gates
class Node {
	int x;
	int y;
	Node(int x, int y){
		this.x = x;
		this.y = y;
	}
}

public void wallsAndGates(int[][] rooms) {

	Deque
	for (int i = 0; i < rooms.length; i++) {
		for (int j = 0; j < rooms[0].length; j++) {
			if (rooms[i][j] == 0) {
				deque.offer(new Node(i, j));
			}
		}
	}
	int level = 0;
	while (!deque.isEmpty()) {
		int size = deque.size();
		level++;
		for (int i = 0; i < size; i++) {
			Node cur = deque.poll();
			int[][] dir = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
			for (int i = 0; i < 4; i++) {
				int x = cur.x + dir[i][0];
				int y = cur.y + dir[i][1];
				if (0 <= x && x < rooms.length && 0 <= y && y < rooms[0].length) {
					if (rooms[x][y] == Integer.MAX_VALUE) {
						rooms[x][y] = level;
						deque.offer(new Node(x, y));
					}
				}
			}
		}
	}

}


// reverse nodes in k-group
public ListNode reverseKGroup(ListNode head, int k){
	if (k == 1 || head == null) {
		return head;
	}
	ListNode dummy = new ListNode(0);
	dummy.next = head;
	ListNode temp  = dummy;

	while (head != null) {
		if (isValid(head, k)) {
			while (k > 1) {
				ListNode next = head.next;
				head.next = next.next;
				next.next = temp.next;
				temp.next = next;
			}
			temp = head;
			head = head.next;
		} else {
			return dummy.next;
		}
	}

	return dummy.next;

}
public boolean isValid (ListNode head, int k) {
	int count = 0;
	while (head != null) {
		count += 1;
		head = head.next;
		if (count >= k) {
			return true;
		}
	}
	return false;
}


// word pattern
public boolean wordPattern(String pattern, String str) {
	String[] strs = str.trim().split(" ");
	if (pattern.length() != strs.length) {
		return false;
	}
	Map<Character, String> map = new HashMap<>();
	Set<String> set = new HashSet<>();
	for (int i = 0; i < pattern.length(); i++) {
		char c = pattern.charAt(i);
		String s = strs[i];
		if (!map.containsKey(c)) {
			if (set.contains(s)) {
				return false;
			} else {
				map.put(c, s);
				set.add(s);
			}
		} else {
			if (!map.get(c).equals(s)) {
				return false;
			} else if (!set.contains(s)) {
				return false;
			}
		}

	}
	return true;
}

//Rearrange a string so that all same characters become d distance away

public String rearrangeString (String s, int n) {

}



//word pattern II 
public boolean wordPatternMatch(String pattern, String str) {

	Map<Character, String> map = new HashMap<>();
	Set<String> set = new HashSet<>();
	boolean result = false;

	helper (pattern, str, 0, 0, result, map, set);

	return result;
}
public void helper (String pattern, String str, int x, int y, ) {
	if (x == pattern.length() && y == str.length()) {
		result = true;
		return;
	}
	if (x >= pattern.length() || y >= str.length()) {
		return;
	}
	char c = pattern.charAt(i);
	for (int j = y + 1; j <= str.length(); j++) {
		String s = str.substring(y, j);
		if (!map.containsKey(c) && !set.contains(s)) {
			map.put(c, s);
			set.add(s);
			helper(pattern, str, x + 1, j);
			map.remove(c);
			set.remove(s);
		} else if (map.containsKey(c) && map.get(c).equals(s)) {
			helper(pattern, str, x + 1, j);
		}

	}

}

// longest increasing subsequence
public int lengthOfLIS(int[] nums) {
	if (nums == null) {
		return 0;
	}
	if (nums.length() <= 1) {
		return nums.length();
	}
	List<Integer> result = new ArrayList<>();
	result.add(nums[0]);
	for (int i = 1; i < nums.length; i++) {
		if (nums[i] < nums[0]) {
			nums[0] = nums[i];
		} else if (nums[i] > nums[i - 1]) {
			result.add(nums[i]);
		} else {
			nums[helper(result, target)] = nums[i];
		}
	}
	return result.size();
}

public int helper (List<Integer> nums, int target) {
	int left = 0;
	int right = nums.length - 1
	while (start + 1 < right) {
		int mid = start + (right - start) / 2;
		if (nums.get(mid) < target) {
			left = mid;
		} else {
			end = mid;
		}
	}
	if (nums.get(left) > target) {
		return start;
	}
	return end;
}



//Russian Doll
public int maxEnvelopes(int[][] envelopes) {

// sort the array base on the width
	Arrays.sort(envelopes, new Comparator<int[] envelope>(){
		public int compare (int[] e1, int[] e2) {
			if (e1[0] == e2[0]) {
				return e1[1] - e1[1];
			} else {
				return e1[0] - e1[1];
			}
		}
	});
	List<Integer> result = new ArrayList<>();
// find the longest increasing base on the height
	for (int i = 0; i < envelopes.length; i++) {
		if (list.size() == 0 || list.get(list.size() - 1) < envelopes[i][1]) {
			list.add(envelopes[i][1]);
		}

		list.set(helper(result, envelopes[i][1]), envelopes[i][1]);
	}
	return list.size();
}
public int helper (List<result> result, int target) {
	int left = 0;
	int right = result.size();
	while (left + 1 < right) {
		int mid = left + (right - left) / 2;
		if (result.get(mid) < target) {
			left = mid;
		} else {
			right = mid;
		}
	}
	if (result.get(start) > target) {
		return start;
	}
	return end;
}


// Unique Word Abbreviation

public class ValidWordAbbr {
    private Set<String> uniqueDict;
    private Map<String, String> abbrDict;
 
    public ValidWordAbbr(String[] dictionary) {
        uniqueDict = new HashSet<>();
        abbrDict = new HashMap<>();
         
        for (String word : dictionary) {
            if (!uniqueDict.contains(word)) {
                String abbr = getAbbr(word);
                if (!abbrDict.containsKey(abbr)) {
                    abbrDict.put(abbr, word);
                } else {
                    abbrDict.put(abbr, "");
                }
                 
                uniqueDict.add(word);
            }
        }
    }
 
    public boolean isUnique(String word) {
        if (word == null || word.length() == 0) {
            return true;
        }
         
        String abbr = getAbbr(word);
        if (!abbrDict.containsKey(abbr) || abbrDict.get(abbr).equals(word)) {
            return true;
        } else {
            return false;
        }
    }
     
    private String getAbbr(String word) {
        if (word == null || word.length() < 3) {
            return word;
        }
         
        StringBuffer sb = new StringBuffer();
        sb.append(word.charAt(0));
        sb.append(word.length() - 2);
        sb.append(word.charAt(word.length() - 1));
         
        return sb.toString();
 
    }
     
}

//Smallest Rectangle Enclosing Black Pixels

public class Solution {

	public int minArea(char[][] image, int x, int y) {
		if (image.length == 0 || image[0].length == 0 || image == null) {
			return 0;
		}
		int rowNum = image.length;
		int colNum = iamge[0].length;

		int left = helper (image, 0, y, 0, rowNum, true, true);
		int right = helper (image, y + 1, colNum, 0, rowNum, false, true);
		int top = helper (image, 0, x, left, right, true, false);
		int down = helper(image, x + 1, rowNum, left, right, false, false);
		return (right - left) * (top - down);
	}
	public int helper (char[][] image, int start, int end, int min, int max,
						boolean searchLow, boolean searchHorizontal) {
		while (start < end) {
			int mid = start + (end - start) / 2;
			hasBleackPixel = false;
			for (int i = min; i < max; i++) {
				char c = searchHorizontal ? image[i][mid] : image[mid][i]; 
				if (c == '1') {
					hasBleackPixel = true;
					break;
				}
			}
			if (hasBleackPixel = searchLow) {
				end = mid;
			} else {
				start = mid;
			}
		}
		return start;
	}
}

//Encode and Decode
public class Codec {
	public String encode(List<String> strs) {
		StringBuilder sb = new StringBuilder();
		for (String str : strs) {
			int len = str.length();
			str.append(String.valueOf(len)).append('#');
			str.append(str);
		}
		return sb.toString();
	}
	public List<String> decode(String s) {
		List<String> result = new ArrayList<>();
		int i = 0;
		for (i < s.length()) {
			int pos = s.indexOf('#', i);
			int len = Integer.parseInt(s.substring(start, pos));
			result.add(s.substring(pos + 1, pos + 1 + len));
			start = pos + 1 + len;
		}
		return result;
	}
}

//Binary Tree Longest Consecutive Sequence
int max = 0;
public int longestConsecutive(TreeNode root) {

	if (root == null) {
		return 0;
	}
	if (root.left == null && root.right == null) {
		return 1;
	}
	int left = longestConsecutive(root.left);
	int right = longestConsecutive(root.right);
	int result = 0;
	if (right == root.val + 1) {
		result = right + 1;
	} else if (left == root.val + 1) {
		result = left + 1;
	}
	max = Math.max(result, max);
	return result;
}

//Maximum Depth of Binary Tree

public int maxDepth(TreeNode root) {
	if (root == null) {
		return 0;
	}
	return helepr(root);

}
public int helper (TreeNode root) {
	if (root == null) {
		return 0;
	}
	int left = helper (root.left);
	int right = helper (roo.right);
	int result = Math.max(left, right) + 1;
	return result;
}


//Factor Combinations
public List<List<Integer>> getFactors(int n) {
	List<List<Integer>> result = new ArrayList<>();
	if (n == 1) {
		return result;
	}
	List<Integer> list = new ArrayList<>();
	helper(2, n, result, list);
	return result;
}
public void helper (int start, int n, List<List<Integer>> result, List<Integer> list) {

	if (n === 1) {
		if (list.size() > 1) {
			result.add();
			return;
		}
	}

	for (int i = start; i <= n; i++) {
		if (n % i == 0) {
			list.add(i);
			helper(i, n / i, result, list);
			list.remove(list.size() - 1);
		}
	}

}



//Text Justification
public ArrayList<String> fullJustify(String[] words, int L) {
	ArrayList<String> result = new ArrayList<>();
	if (words.length == 0 || words == null || L == 0) {
		return result;
	}

	int preSize = 0;
	int start = 0;
	
	for (int i = 0; i < words.length; i++) {
		String word = words[i];
		
		if ((preSize + (i - start) + word.length()) > L) {
			int spaceNum = 0;
			int spaceLeft = 0;

			if (i - start - 1 > 0) {
				spaceNum = (L - (preSize)) / (i - start - 1);
				spaceLeft = (L - (preSize)) % (i - start - 1);
			}
			
			StringBuilder sb = new StringBuilder();
			for (int j = start; j < i; j++) {
				sb.append(words[j]);
				if (j < i - 1) {
					for (int k = 0; k < spaceNum; k++) {
						sb.append(" ");
					}
					if (spaceLeft > 0) {
						sb.append(" ");
						spaceLeft--;
					}
				}		
			}
			for (int j = sb.length(); j < L; j++) {
				sb.append(" ");
			}
			result.add(sb.toString());
			start = i;


		}
		preSize += word.length();
	}

	StringBuilder sb = new StringBuilder();
	for (int i = start; i < words.length; i++) {
		sb.append(words[start]);
		if (sb.length() < L) {
			sb.append(" ");
		}
	}
	for (int i = sb.length(); i < L; i++) {
		sb.append(" ");
	}
	result.add(sb.toString());
	return result;

}


//Design Phone Directory
public class PhoneDirectory {
	Bitset bitset;
	int max;
	int smallestFreeIndex;
	public PhoneDirectory(int maxNumbers) {
		this.bitset = new Bitset(maxNumbers);
		this.max = maxNumbers
	}
	public int get() {
		if (smallestFreeIndex == max) {
			return -1;
		}
		int num = smallestFreeIndex;
		bitset.set(smallestFreeIndex);
		smallestFreeIndex = bitset.nextClearBit(smallestFreeIndex);
		return num;
	}
	public boolean check(int number) {
		return bit.get(number) == false;
	}
	public void release(int number) {
		
	}
}


public int[] getModifiedArray(int length, int[][] updates) {
	
	int[] result = new ArrayList<>();
	for (int i = 0; i < updates.length; i++) {
		int start = updates[i][0];
		int end = updates[i][1];
		int inc = updates[i][2];
		result[start] += inc;
		if (end + 1 <= length - 1) {
			result[end + 1] -= inc;
		}
	}
	int sum = 0;
	for (int i = 0; i < result.length; i++) {
		sum += result[i];
		result[i] = sum;
	}
	return result;
}



//Find Leaves of Binary Tree
public List<List<Integer>> findLeaves(TreeNode root) {

	List<List<Integer>> result = new ArrayList<>();
	helper(result, root);
	return result;
}
public int helper (List<List<Integer>> result, TreeNode root) {
	if (root == null) {
		return -1;
	}
	int left = helper(result, root.left);
	int right = helper(result, root.right);
	int cur = Math.max(left, right) + 1;
	if (result.size() <= cur) {
		List<Integer> list = new ArrayList<>();
		result.add(list);
	}
	result.get(cur).add(root.val);
	return cur;
}


// plusOne ListNode
public ListNode plusOne(ListNode head) {
	TrieNode newHead = reverse(head);

	ListNode temp = new newHead;
	while (temp != null) {
		if (temp.val + 1 <= 9) {
			p.val = p.val + 1
			break;
		} else {
			p.val = 0;
			if (p.next == null) {
				p.next = new ListNode(1);
				break;
			}
			p = p.next;
		}
	}
	return reverse(newHead);
}
public ListNode reverse(ListNode head) {
	ListNode pre = null;
	while (head != null) {
		ListNode next = head.next;
		head.next = pre;
		pre = pre.next;
		head = next;
	}
	return pre;
}






calss TrieNode {
	List<Integer> list;
	int index;
	TrieNode[] children;
	TrieNode() {
		this.index = -1;
		this.list = new ArrayList<>();
		this.children = new TrieNode[26];
	}
} 
public List<List<Integer>> palindromePairs(String[] words) {
	List<List<Integer>> result = new ArrayList<>();

	if (words.length == 0 || words == null) {
		return result;
	}
	TrieNode root = buildTree(words);
	helper(result, root, words);
	return result;
}
public void helper (List<List<Integer>> result, TrieNode root, String[] words) {

	for (int i = 0; i < words.length; i++) {
		String word = words[i];
		TreeNode node = root;
		for (int j = 0; j < word.length(); j++) {
			char c = word.charAt(j);
			if (root.index >= 0 && root.index != j 
				&& isPalindrome(word, j, word.length() - 1)) {
				node = node.children[c - 'a'];
			}
			if (root == null) {
				return;
			}
		}

		for (int j : node.list) {
			if (i == j) {
				continue;
			} 
			result.add(Arrays.asList(i, j));
		}
	}

}

public TrieNode buildTree (String[] words) {
	TrieNode root = new TrieNode();
	for (int i = 0; i < words.length; i++) {
		TrieNode node = root;
		String word = words[i];
		for (int j = word.length() - 1; j >= 0; j--) {
			char c = word.charAt(j);
			if (node.children[c - 'a'] == null) {
				node.children[c - 'a'] = new TrieNode();
			}
			
			if (isPalindrome(word, 0, j)) {
				node.list.add(i);
			}
			node = node.children[c - 'a'];
		}
		node.index = i;
		node.list.add(index);
	}
	return root;
}
public boolean isPalindrome (String str, int left, int right) {
	while (left < right) {
		if (s.charAt(left) != s.charAt(right)) {
			return false;
		}
		left++;
		right--;
	}
	return true;
}






public class Solution {
    public void recoverTree(TreeNode root) {
        TreeNode current = root;
        TreeNode pre = null;
        TreeNode node1 = null;
        TreeNode node2 = null;
        while (current != null) {
        	if (current.left != null) {
        		TreeNode temp = current.left;
        		while (temp.right !== null && temp.right != current) {
        			temp = temp.right;
        		}
        		if (temp.right == null) {
        			temp.right = current;
        			current = current.left;
        		} else {
        			t.right = null;
        			if (pre != null) {
        				if (pre.val >= root.val) {
        					if (first == null) {
        						first = pre;
        					}
        					second = current;
        				}
        			}
        			pre = current;
        			current = current.right;
        		}
        	} else {
        		if (pre != null) {
        			if (pre.val >= root.val) {
        				if (first == null) {
        					first = pre;
        				}
        				second = current;
        			}
        		}
        		pre = current;
        		current = current.right;
        	}
        } 

        while (current != null) {
            if (current.left == null) {
                if (prev != null) {
                    if (prev.val >= current.val) {
                        if (node1 == null)
                            node1 = prev;
                        node2 = current;
                    }
                }
                prev = current;
                current = current.right;
            } else {
                TreeNode t = current.left;
                while (t.right != null && t.right != current) {
                    t = t.right;
                }
                if (t.right == null) {
                    t.right = current;
                    current = current.left;
                } else {
                    t.right = null;
                    if (prev != null) {
                        if (prev.val >= current.val) {
                            if (node1 == null)
                                node1 = prev;
                            node2 = current;
                        }
                    }
                    prev = current;
                    current = current.right;
                }
            }
        }
        int tmp = node1.val;
        node1.val = node2.val;
        node2.val = tmp;
    }
}

if (node.left == null) {
	if (pre != null) {
		if (pre.val >= cur.val) {
			if (first == null) {
				first = pre;
			}
			second = cur;
		}
	}
	pre = cur;
	cur = cur.right;
} else {

}

public class Solution {
    
    TreeNode first = null;
    TreeNode second = null;
    TreeNode pre = null;
    public void recoverTree(TreeNode root) {
    	if (root == null) {
    		return;
    	}
    	
    	if (second != null && first != null) {
    		swap(first, second);
    	}
    	
    }
    public void helper (TreeNode root) {
    	if (root == null) {
    		return;
    	}
    	helper(root.left);
    	if (pre != null) {
    		if (root.val <= pre.val) {
    			if (first == null) {
    				first = pre;
    			}
    			second = root;
    		}
    	}
    	pre = root;
    	helper(root.right);
    }
    public void swap (TreeNode n1, TreeNode n2) {
    	int temp = n1.val;
    	n1.val = n2.val;
    	n2.val = temp;
    }
}

public class ZigzagIterator implements Iterator<Integer> {
	List<Iterator<Integer>> list; 
	int turn;
	public ZigzagIterator(List<Iterator<Integer>> list) {
		this.list = new LinkedList<>();
		for (Iterator<Integer> i : list) {
			if (i.hasNext()) {
				list.add(i);
			}
		}
		this.turn = 0;
	}
	public Integer next() {
		if (!hasNext()) {
			return 0;
		}
		Integer result = 0;
		int pos = turn % list.size();
		Iterator<Integer> cur = list.get(pos);
		result = cur.next();
		if (!cur.hasNext()) {
			list.remove(turn % list.size());
			turn = pos - 1;
		}
		turn++;
		return result;
	}
	public boolean hasNext() {
		return list.size() > 0;
	}
}


public class ZigzagIterator {
	int cur;
	Iterator<Integer> i1;
	Iterator<Integer> i2;
	public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
		this.i1 = v1.iterator();
		this.i2 = v2.iterator();
		this.cur = 0;
	}
	public int next() {
		if (!hasNext()) {
			return 0;
		}
		cur++;
		if ((cur % 2 == 1 && i1.hasNext()) || (!i2.hasNext())) {
			return i1.next();
		} else if ((cur % 2 == 0 && i2.hasNext()) || (!i1.hasNext())) {
			return i2.next();
		} 
		return 0;
	}
	public boolean hasNext() {
		return i1.hasNext() || i2.hasNext();
	}
}


public class Vector2D {
	List<Iterator<Integer>> list;
	int cur = 0;
	public Vector2D(List<List<Integer>> vec2d) {
		this.list = new ArrayList<Iterator<Integer>>();
		for (List<Integer> l : vec2d) {
			if (l.size() > 0) {
				this.list.add(l.iterator());
			}
		}
	}
	public int next() {
		Integer res = list.get(cur).next();
		if (!list.get(cur).hasNext()) {
			cur++;
		}
		return res;
	}
	public boolean hasNext() {
		return cur < list.size() && list.get(cur).hasNext();
	}
}


public String rearrangeString(String str, int k) {

	if (k == 0) {
		return str;
	}

	Map<Character, Integer> map = new HashMap<>();
	for (int i = 0; i < str.length; i++) {
		char c = str.charAt(i);
		map.put(c, map.containsKey(c) ? 1 : map.get(c) + 1);
	}

	PriorityQueue<Character> heap = new PriorityQueue<Character>(map.size(), new Comparator<Character>(){
		public int compare (Character c1, Character c2) {
			return map.get(c2) - map.get(c1);
		}
	});

	for (Character c : map.keySet()) {
		heap.offer(c);
	}
	char[] s = new char[str.length()];
	Arrays.fill(s, '\0');

	for (int i = 0; i < s.length; i++) {
		int p = i;
		while (s[p] != '\0') {
			p++;
		}
		char c = heap.poll();
		int count = map.get(c);
		for (int i = 0; i < count; i++) {
			if (p >= n) {
				return "";
			}
			s[p] = c;
			p += k;
		}

	}
	return new String(s);
}


class Hit {
	int timestamp;
	int count;
	Hit next;
	Hit (int timestamp) {
		this.timestamp = timestamp;
		this.count = 1;
	}
}
public HitCounter {
	Hit start;
	Hit tail;
	int count;
	public HitCounter() {
		this.start = new Hit(0);
		this.tail = start;
		this.count = 0;
	}
	public void hit (int timestamp) {
		if (tail.timestamp == timestamp) {
			tail.count++;
			count++;
		} else {
			tail.next = new Hit(timestamp);
			tail = tail.next;
			count++;
		}
		getHits(timestamp);
	}
	public int getHits (int timestamp) {
		while (start.next != null && timestamp - start.next.timestamp >= 300) {
			count -= start.next.count;
			start.next = start.next.next;
		}
		if (start.next == null) {
			tail = start;
		}
		return count;
	}

}


public int[] sortTransformedArray(int[] nums, int a, int b, int c) {

	int[] result = new int[nums.length];

	int start = 0;
	int end = nums.length - 1;

	int nextIndex = 0;

	if (a > 0 || (a == 0 && b > 0)) {
		nextIndex = nums.length - 1
	}
	if (a < 0 || (a == 0 && b < 0)) {
		nexIndex = 0;
	}

	double mid = -1 * (b * 1.0) / (2.0 * a);

	while (start <= end) {
		if (a > 0 || (a == 0 && b > 0)) {
			if (Math.abs(mid - nums[start]) > Math.abs(nums[end] - mid)) {
				int x = nums[start++];
				result[nextIndex--] = a * x * x + b * x + c;
			} else {
				int x = nums[end--];
				result[nextIndex--] = a * x * x + b * x + c;
			}
		} else if (a < 0 || (a == 0 && b < 0) {
			if (Math.abs(mid - nums[start]) > Math.abs(nums[end] - mid)) {
				int x = nums[start++];
				result[nextIndex++] = a * x * x + b * x + c;
			} else {
				int x = nums[end--];
				result[nextIndex++] = a * x * x + b * x + c;;
			}
		}
	}
	return result;

}

public int findNthNum (int n) {
	
	long len = 1;
	int start = 1;
	int cnt = 9;
	while (n > len * cnt) {
		n -= len * cnt;
		len += 1;
		cnt *= 10;
		start *= 10;
	}
	start += (n - 1) / len;
	String str = String.valueOf(start);
	return str.charAt((n - 1) % len) - '0';
}


public boolean shouldPrintMessage(int timestamp, String message) {
	Map<String, Integer> map = new HashMap<>();
	if (!map.containsKey(message)) {

	}
}

class Node {
	int val;
	int pos;
	Node(int val, int pos) {
		this.val = val;
		this.pos = pos;
	} 
}

public List<List<Integer>> verticalOrder(TreeNode root) {
	List<List<Integer>> result = new ArrayList<>();
	if (root == null) {
		return result;
	}
	Map<Integer, List<Integer>> map = new HashMap<>();
	map.put(0, root.val);
	Deque<Node> deque = new LinkedList<>();
	deque.offer(new Node(root.val, 0));

	int minLevel = 0;
	int maxLevel = 0;
	while (!deque.isEmpty()) {
		
		Node cur = deque.poll();
		minLevel = Math.min(minLevel, cur.pos - 1);
		maxLevel = Math.max(maxLevel, cur.pos + 1);
		if (cur.left != null) {
			if (!map.containsKey(cur.pos - 1)) {
				List<Integer> list = new ArrayList<>();
				list.add(cur.left.val);
				map.put(cur.pos - 1, list);
			} else {
				map.put(cur.pos - 1, map.get(cur.pos - 1).add(cur.left.val));
			}
			deque.offer(new Node(cur.left, cur.pos - 1));
		}
		if (cur.right != null) {

		}
	}
	for (int i = minLevel; i <= maxLevel; i++) {
		if (map.containsKey(i)) {
			result.add(map.get(i));
		}
	}
	return result;
}




class MedianFinder {
	PriorityQueue<Double> min;
	PriorityQueue<Double> max;
	public MedianFinder()	{	
		this.min = new PriorityQueue<>();
		this.max = new PriorityQueue<>(11, new Comparator<>(Double){
			public int compare (Double i1, Double i2) {
				return i2 - i1;
			}
		})
	}

	public void addNum(int num) {
		double cur = (double)num;
		if (max.isEmpty()) {
			max.offer(cur);
		} else if (Math.abs(max.size() - min.size()) >= 1) {
			if (cur >= max.peek()) {
				if (min.size() > max.size()) {
					max.offer(min.pop());
					min.offer(cur);
				} else {
					min.offer(cur);
				}
			} else if (cur < max.peek()) {
				if (min.size() > max.size()) {
					min.offer(max.pop());
					max.offer(cur);
				} else {
					min.offer(cur);
				}
			}
		} else if (cur >= min.peek()) {
			min.offer(cur);
		} else if (cur <= max.peek()) {
			max.offer(cur);
		}
	}
	public double findMedian() {
		if (min.size() > max.size()) {
			return min.peek();
		} else if (min.size() < max.size()) {
			return max.peek();
		}
		
		return (min.peek() + max.peek()) / 2;
		
	}
}

public NestedInteger deserialize(String s) {
        
	Stack<NestedInteger> stack = new Stack<>();
	StringBuilder sb = new StringBuilder();
	NestedInteger result = new NestedInteger();
	for (int i = 0; i < s.length(); i++) {
		
		char c = s.charAt(i);
		if (c == '[') {	
			NestedInteger cur = new NestedInteger();
			stack.push(cur);
		} else if (c == ']') {
			if (sb.length() != 0) {
				NestedInteger cur = new NestedInteger(Integer.parseInt(sb.toString)));
				stack.peek().add(cur);
				sb.setLength(0);
			}
			NestedInteger top = stack.pop();
			if (!stack.isEmpty()) {
				stack.peek().add(top);
			} else {
				return top;
			}
			
		} else if (c == ',') {

			NestedInteger cur = new NestedInteger(Integer.parseInt(sb.toString)));
			stack.peek().add(cur);
			sb.setLength(0);
			
		} else {
			s.append(c);
		}
	}
	if (sb.length() != 0) {
		return new NestedInteger(Integer.parseInt(sb.toString()));
	}
	return null;
}


public class NestedIterator implements Iterator<Integer> {
	Deque<NestedInteger> deque;
    public NestedIterator(List<NestedInteger> nestedList) {
        this.deque = new LinkedList<NestedInteger>();
  
        helper(nestedList, deque);
    	
    }
    public void helper (LinkedList<NestedInteger> nestedList, Deque<NestedInteger> deque) {
        
        for (NestedInteger n : nestedList) {
        	if (n.isInteger()) {
        		deque.offer(n);
        	} else {
        		helepr(n.getList(), deque);
        	}
        }
    } 
    @Override
    public Integer next() {
        if (hasNext()) {
        	return deque.poll();
        }
    }

    @Override
    public boolean hasNext() {
        return !deque.isEmpty();
    }
}


public int depthSumInverse(List<NestedInteger> nestedList) {
	Map<Integer, Integer> map = new HashMap<>();
	helper(nestedList, depth, map);
	
	int result = 0;
	int maxDepth = Integer.MIN_VALUE;
	for (int n : map.keSet()) {
		maxDepth = Math.max(maxDepth, n);
	}

	for (int n : map.keySet()) {
		result += (maxDepth - n + 1) * map.get(n);
	}

	return result;

}
public void helper (List<NestedInteger> nestedList, int depth, Map<Integer, Integer> map) {
	
	for (NestedInteger n : nestedList) {
		if (n.isInteger()) {
			Integer num = map.get(n);
			if (num == null) {
				map.put(depth, n.getInteger());
			} else {
				map.put(depth, map.get(depth) + n.getInteger());
			}
		} else {
			helper(n.getList(), depth + 1);
		}
	}
	return max;
}

public List<Integer> closestKValues(TreeNode root, double target, int k) {

	List<Integer> result = new ArrayList<>();
	if (root == null) {
		return result;
	}
	Stack<Integer> precedessor = new Stack<>();
	Stack<Integer> successor = new Stack<>();
	
	getPredecessor(root, target, precedessor);
	getSuccessor(root, target, successor);

	for (int i = 0; i < k; i++) {
		if (precedessor.isEmpty()) {
			result.add(successor.pop());
		} else if (successor.isEmpty()) {
			result.add(precedessor.pop());
		} else if (Math.abs((double) precedessor.peek() - target) < Math.abs((double) successor.peek) - target) {
			result.add(precedessor.pop());
		} else {
			result.add(successor.pop());
		}
	}
	return result;

}
private void getPredecessor(TreeNode root, double target, Stack<Integer> precedessor) {
	if (root == null) {
		return;
	}
	getPredecessor(root.left, target, precedessor);

	if (root.val > target) {
		return;
	}
	precedessor.push(root.val);
	getPredecessor(root.right, target, precedessor);

}
private void getSuccessor(TreeNode root, double target, Stack<Integer> successor) {
	if (root == null) {
		return;
	}
	getSuccessor(root.right, target, successor);
	if (root.val <= target) {
		return;
	}
	successor.push(root.val);
	getSuccessor(root.left, target, successor);
}

public int closestValue(TreeNode root, double target) {

	int result = root.val;
	while (root != null) {
		result = Math.abs(result - target) < Math.abs(root.val - target) ? result : root.val;
		root = target < root.val ? root.left : root.right;
	}
	return result;

}

public solution {
	int result = 0;
	public int countUnivalSubtrees(TreeNode root) {  

		helper(TreeNode root);
		return result;
	}

	public boolean helper (TreeNode root) {
		if (root == null) {
			return true;
		}
		if (root.left == null && root.right == null) {
			result++;
			return true;
		}
		boolean left = helper(root.left);
		boolean right = helepr (root.right);
		if (left && right 
			&& (root.left == null || root.left.val == root.val) 
			&& (root.right == null || root.right.val == root.val)) {
			result++;
			return true;
		}
		return false;

	}
}

public void connect(TreeLinkNode root) {
	if (root == null) {
		return;
	}

	TreeLinkNode head = root;
	while (head != null) {
		TreeLinkNode cur = new TreeLinkNode(0);
		TreeLinkNode temp = cur;

		while (head != null) {
			if (cur.left != null) {
				temp.left.next = temp.right;
			}
			if (temp.right != null && temp.next != null) {
				temp.right.next = temp.next.left;
			}
			temp = temp.next;
		}

		cur = cur.left;
	}

}


public boolean isBalanced(TreeNode root) {

	if (root == null) {
		return true;
	}
	if (getHeight(root) == -1) {
		return false;
	}
	return true;
}
public int getHeight (TreeNode root) {
	if (root == null) {
		return 0;
	}
	int left = getHeight(root.left);
	int right = getHeight(root.right);
	if (left == -1 || right == -1) {
		return -1;
	}
	if (Math.abs(left - right) > 1) {
		return -1;
	}
	return Math.max(left, right) + 1;
}


class Node {
	Node pre;
	Node next;
	int val;
	List<Integer> set;
	Node (int val) {
		this.set = new ArrayList<>();
		this.pre = null;
		this.next = null;
		this.val = val;
	}
}
class item {
	int key;
	int val;
	Node parent;
	public item(int key, int val, Node parent) {
		this.key = key;
		this.val = val;
		this.parent = parent;
	}
}

int capacity;
Map<Integer, Integer> map;
Node head;
Node tail;
public LFUCache(int capacity) {
	this.capacity = capacity;
	this.map = new HashMap<>();
	this.head = new Node(-1);
	this.tail = new Node(-1);
	head.next = tail;
	tail.pre = head;
}

public void set(int key, int value) {
	if (get(key) != -1) {
		map.get(key).val = val;
		return;
	}
	if (map.size() == capacity) {
		getLFUitem();
	}
	Node newpar = head.next;
	if (newpar.val != 1) {
		newpar = getNewNode(1, head, newpar);
	}

	item curItem = new intem(key, val, newpar);
	map.put(key, curItem);
	newpar.set.add(key);
	return;
}

public int get(int key) {
	
	if (!map.containsKey(key)) {
		return -1;
	} 
	item cur = map.get(key);
	Node curpar = cur.parent;
	if (curpar.next.val == curpar.val + 1) {
		cur.parent = curpar.next;
		cur.parent.set.add(key);
	} else {
		Node newpar = getNewNode(curpar.val + 1, curpar, curpar.next);
		cur.parent = newpar;
		newpar.set.add(key);
	}
	curpar.set.remove(new Integer(key));
	if (curpar.set.isEmpty()) {
		deleteNode(curpar);
	}

	return cur.val;
}



public Node getNewNode (int val, Node pre, Node next) {
	Node temp = new Node(val);
	temp.pre = pre;
	temp.next = next;
	pre.next = temp;
	next.pre = temp;
	return temp;
}

public void getLFUitem () {
	Node temp = head.next;
	int LFUkey = temp.set.get(0);
	temp.set.remove(0);
	map.remove(LFUkey);
	if (temp.set.isEmpty()) {
		deleteNode(temp);
	}
	return;
}
public void deleteNode (Node temp) {
	temp.pre.next = temp.next;
	temp.next.pre = temp.pre;
	return;
}

0 2 1 0 1 
pos
2 1 1 0 0
  
public void moveZeros (int[] nums) {
	
	int pos = -1;
	for (int i = 0; i < nums.length; i++) {
		if (pos == -1 && nums[i] == 0) {
			pos = i;
		} else if (pos != -1 && nums[i] != 0) {
			swap(nums, pos, i);
			pos++;
		}
	}

}

public int divide(int dividend, int divisor) {

	if (divisor == 0) {
		return Integer.MAX_VALUE;
	} else if (dividend == Integer.MAX_VALUE && (divisor == 1 || divisor == - 1)) {
		return divisor == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
	} else if (divident == Integer.MIN_VALUE && (divisor == -1 || divisor == 1)) {
		return divisor == 1 ? Integer.MIN_VALUE : Integer.MAX_VALUE;
	} else if (dividend == 0) {
		return 0;
	}
	boolean pos = true;
	if ((divisor < 0 && dividend > 0) || (divisor > 0 && dividend < 0)) {
		pos = false;
	}
	Long a = Math.abs((long)dividend);
	Long b = Math.abs((long)divisor);
	int result = 0;
	
	while (a >= b) {
		int shift = 0;
		while (a >= (b << shift)) {
			shift++;
		}
		a -= b << (shift - 1);
		result += a << (shift - 1);
	}
	return pos ? result : -result;
}

public List<String> removeInvalidParentheses(String s) {

	List<String> result = new ArrayList<>();
	char[] temp = new char[]{'(', ')'};
	helepr (result, temp, s, 0, 0);
	return result;
}

public void helper (List<String> result, char[] temp, String s, int index_i, int index_j) {
	int cur;
	for (int i = index_i; i < s.length(); i++) {
		if (s.charAt(i) == temp[0]) {
			cur++;
		}
		if (s.charAt(i) == temp[1])  {
			cur--;
		}
		if (cur >= 0) {
			continue;
		}
		for (int j = index_j; j <= i; j++) {
			if (s.charAt(j) == temp[1] && 
				(j == index_j || s.charAt(j - 1) != temp[1])) {
				helper(result, temp, s.substring(0, j) + s.substring(j + 1, s.length()), i, j);
			}
		}
		return;
	}

	String reversed = new StringBuilder(s).reverse().toString();
	if (temp[0] == '(') {
		helper(result, new char[]{'(', ')'}, reversed, 0, 0);
	} else {
		result.add(reversed);
	}

}




Map<Character, String> map;
Set<String> set;
boolean result;

public boolean wordPatternMatch(String pattern, String str) {
	map = new HashMap<>();
	set = new HashSet<>();
	result = false;
	helper (pattern, str, 0, 0);
	return result;
}
public void helper (String pattern, String str, int i, int j) {
	if (i == pattern.length() && j = str.length()) {
		result = true;
		return;
	}
	if (i >= pattern.length() || j >= str.length()) {
		return;
	}
	char c = pattern.charAt(i);

	for (int cut = j + 1; cut <= str.length(); cut++) {
		String substr = str.substring(j, cut);
		if (!set.contains(substr) && !map.containsKey(c)) {
			map.put(c, substr);
			set.add(substr);
			helper (pattern, str, i + 1, cut);
			map.remove(c);
			set.remove(substr);
		} else if (map.containsKey(c) && map.get(c).equals(substr)) {
			helper(pattern, str, i + 1, cut);
		}
	}

}

public int shortestDistance(int[][] grid) {




}




public class LFUCache {

	public LFUCache(int capacity) {

	}
	
	public void set(int key, int value) {

	}

	public int get(int key) {

	}

}



public int countPrimes(int n) {
	boolean[] primes = new boolean[n];
	Arrays.fill(primes, true);
	for (int i = 2; i < n; i++) {
		if (primes[i]) {
			for (int j = i * 2; j < n; j = j + i) {
				primes[j] = false;
			}
		}
	}
	int count = 0;
	for (int i = 2; i < n; i++) {
		if (primes[i]) {
			count++;
		}
	}
	return count;
}



public List<String> sol (String s) {
	List<String> result = new ArrayList<>();
	if (s.length() == 0 || s == null) {
		return result;
	}
	helper (result, s, "");
	return result;
}
public void helper (List<String> result, String s, String temp) {
	if (temp.length() == s.length()) {
		result.add(new String(temp));
		return;
	}
	for (int i = 0; i < s.length(); i++) {
		if (i != 0 && s.charAt(i) == s.charAt(i - 1)) {
			continue;
		}
		temp += "" + s.charAt(i);
		helper (result, s, index + 1, temp);
		temp.substring(0, temp.length() - 1);
	}


}
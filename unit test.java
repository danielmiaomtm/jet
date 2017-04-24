public class Math {
    int a, b;
    Math(int a, int b) {
        this.a = a;
        this.b = b;
    }
    public int add() {
        return a + b;
    }
}




import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
public class MathTest {
    Math math;
    @Before
    public void setUp() throws Exception {
        math = new Math(7, 10);
    }
    @Test
    public void testAdd() {
        Assert.assertEquals(17, math.add());
    }
}

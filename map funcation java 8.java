package test; 

import java.util.ArrayList; 
import java.util.Arrays; 
import java.util.List; 
import java.util.stream.Collectors; 
import static java.util.stream.Collectors.toList; 
/** * Java 8 example to convert each element of List into upper case. You can use * Map function of Java 8 to transform each 
element of List or any collection. * @author Javin */ 

public class Java8MapExample { 
  public static void main(String args[]) { 
    List<String> cities = Arrays.asList("London", "HongKong", "Paris", "NewYork"); 
    System.out.println("Original list : " + cities); 
    System.out.println("list transformed using Java 8 :" + transform(cities));
    
    List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9);
    List<Integer> squares = numbers.stream().map( i -> i*i).collect(Collectors.toList()); 
    
    } 
   
    public static List<String> transform(List<String> listOfString) {
        return listOfString.stream() // Convert list to Stream
                .map(String::toUpperCase) // Convert each element to upper case
                .collect(toList()); // Collect results into a new list
    }





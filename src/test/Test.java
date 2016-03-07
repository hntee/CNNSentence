package test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class Test {

	
	public static void main(String[] args) throws IOException {
		Stream<Path> paths = Files.walk(Paths.get("."));
		String delimiter;
		String os = System.getProperty("os.name");
		if (os.contains("Windows")) {
			delimiter = "\\\\";
		} else {
			delimiter = "/";
		}
		
		paths.filter(p -> p.toString().split(delimiter).length > 4)
        .map(p -> p.toString())
        .forEach(System.out::println);
	}
}

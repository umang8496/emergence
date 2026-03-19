import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class FileProcessor {

    public static String readFile(String filePath) {
        String data = "";
        try {
            BufferedReader br = new BufferedReader(new FileReader(filePath));
            String line;

            while ((line = br.readLine()) != null) {
                data += line; // Inefficient string concatenation
            }

            // Resource leak: not closing BufferedReader
        } catch (IOException e) {
            e.printStackTrace(); // Poor error handling
        }

        return data;
    }

    public static void processData(String input) {
        char[] buffer = new char[10];

        // Potential buffer overflow scenario
        for (int i = 0; i < input.length(); i++) {
            buffer[i] = input.charAt(i); // No bounds check
        }

        System.out.println("Processed data: " + new String(buffer));
    }
}

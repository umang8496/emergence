import java.util.HashMap;
import java.util.Map;

public class UserService {

    private Map<String, String> users = new HashMap<>();

    public void registerUser(String username, String password) {
        // No validation
        users.put(username, password);
    }

    public boolean login(String username, String password) {
        // Null pointer risk
        if (users.get(username).equals(password)) {
            return true;
        }
        return false;
    }

    public String getPassword(String username) {
        // Security issue: exposing password directly
        return users.get(username);
    }

    public void printAllUsers() {
        for (String key : users.keySet()) {
            System.out.println("User: " + key + ", Password: " + users.get(key));
        }
    }
}

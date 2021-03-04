package core.queryexpansion;

import java.util.Set;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import org.json.*;


public class BARTezExpander{

    public static String sendPOST(String text, Set<String> words){
        String words_param = "";
        for (String word : words) {
            words_param = words_param + " " + word;
        }
        try {
            URL obj = new URL("http://localhost:5000/embeddings");
            HttpURLConnection con = (HttpURLConnection) obj.openConnection();
            con.setRequestMethod("POST");

            con.setDoOutput(true);

            String POST_PARAMS = "utterances=" + text + ",keywords=" + words_param;
            try (OutputStream os = con.getOutputStream()) {
                byte[] input = POST_PARAMS.getBytes("utf-8");
                os.write(input, 0, input.length);
            }
            int responseCode = con.getResponseCode();
            System.out.println("POST Response Code :: " + responseCode);

            if (responseCode == HttpURLConnection.HTTP_OK) { //success
                BufferedReader in = new BufferedReader(new InputStreamReader(
                        con.getInputStream()));
                String inputLine;
                StringBuffer response = new StringBuffer();

                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();

                JSONObject job = new JSONObject(response.toString());
                return job.get("cluster1") + "##" + job.get("cluster2") + "##" + job.get("cluster3");
            } else {
                System.out.println("POST request not worked");

            }
        }catch(Exception ignored){}
        return words_param +"##";
    }
}

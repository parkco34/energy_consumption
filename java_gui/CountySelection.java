import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;
import java.util.List;
import java.util.ArrayList;
import org.json.simple.*;
import org.json.simple.parser.*;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class CountySelection {
    private JFrame frame;
    private JList<String> countyList;
    private DefaultListModel<String> listModel;
    private JButton submitButton;
    private JTextArea outputArea;
    private Map<String, String> countyToFIPS;

    public CountySelection() {
        frame = new JFrame("Select Counties");
        frame.setSize(500, 400);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());

        listModel = new DefaultListModel<>();
        countyToFIPS = new HashMap<>();

        // Load county data from JSON
        loadCountyData("county_fips.json");

        countyList = new JList<>(listModel);
        countyList.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
        JScrollPane scrollPane = new JScrollPane(countyList);

        submitButton = new JButton("Get FIPS Codes");
        submitButton.addActionListener(e -> {
            List<String> selectedCounties = countyList.getSelectedValuesList();
            List<String> fipsCodes = new ArrayList<>();

            for (String county : selectedCounties) {
                fipsCodes.add(countyToFIPS.get(county));
            }

            // Format output nicely
            String formattedFIPS = fipsCodes.isEmpty() ? "None Selected" : String.join(", ", fipsCodes);
            outputArea.setText("Selected FIPS Codes: " + formattedFIPS);
        });

        outputArea = new JTextArea();
        outputArea.setEditable(false);
        outputArea.setLineWrap(true);
        outputArea.setWrapStyleWord(true);

        frame.add(scrollPane, BorderLayout.CENTER);
        frame.add(submitButton, BorderLayout.SOUTH);
        frame.add(outputArea, BorderLayout.NORTH);

        frame.setVisible(true);
    }

    private void loadCountyData(String filePath) {
        try {
            JSONParser parser = new JSONParser();
            JSONArray jsonArray = (JSONArray) parser.parse(new FileReader(filePath));

            for (Object obj : jsonArray) {
                JSONObject jsonObject = (JSONObject) obj;
                String county = (String) jsonObject.get("county_name");
                String fips = String.valueOf(jsonObject.get("full_fips"));

                countyToFIPS.put(county, fips);
                listModel.addElement(county);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(CountySelection::new);
    }
}


import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;
import org.json.simple.*;
import org.json.simple.parser.*;

public class CountySelectionGUI {
    private JFrame frame;
    private JList<String> countyList;
    private DefaultListModel<String> listModel;
    private JButton submitButton;
    private JTextArea outputArea;
    private Map<String, String> countyToFIPS;

    public CountySelectionGUI() {
        frame = new JFrame("Select Counties");
        frame.setSize(400, 400);
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
        submitButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                List<String> selectedCounties = countyList.getSelectedValuesList();
                List<String> fipsCodes = new ArrayList<>();
                
                for (String county : selectedCounties) {
                    fipsCodes.add(countyToFIPS.get(county));
                }
                
                outputArea.setText("Selected FIPS Codes: " + fipsCodes);
            }
        });

        outputArea = new JTextArea();
        outputArea.setEditable(false);

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
        SwingUtilities.invokeLater(() -> new CountySelectionGUI());
    }
}

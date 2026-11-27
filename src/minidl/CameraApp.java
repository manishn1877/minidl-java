package minidl;

/**
 * "Start camera" from the command line: starts the web server (every trained model it finds; finger tracking needs none) and
 * opens the camera page in your browser.
 *
 *   java -jar minidl.jar camera [port=8080]
 *
 * The browser asks for permission to use the camera (the JDK itself has no webcam API, so the browser reads the camera);
 * every picture is analysed by the Java program in this process (HandTracker). Nothing leaves your computer.
 */
public final class CameraApp {
    private CameraApp() {}

    public static void main(String[] args) throws Exception {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 8080;
        HandwritingServer.main(new String[]{"auto", String.valueOf(port), "."});
        String url = "http://localhost:" + port + "/camera";
        try {
            if (java.awt.GraphicsEnvironment.isHeadless() || !java.awt.Desktop.isDesktopSupported() || !java.awt.Desktop.getDesktop().isSupported(java.awt.Desktop.Action.BROWSE)) throw new UnsupportedOperationException();
            java.awt.Desktop.getDesktop().browse(java.net.URI.create(url));
            System.out.println("opened " + url + " in your browser: press Start camera and allow the camera");
        } catch (Exception | Error e) {
            System.out.println("open this address in your browser: " + url + "   (then press Start camera and allow the camera)");
        }
    }
}

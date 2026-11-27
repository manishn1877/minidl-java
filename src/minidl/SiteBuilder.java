package minidl;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

/**
 * Builds the GitHub Pages site into docs/ : the handwriting page (the same file the Java server serves), the training chart, and
 * a ".nojekyll" marker so GitHub serves the files as they are.  Turn on Pages in the repository settings: branch main, folder /docs.
 * The page on Pages only draws; it talks to the Java server running on the visitor's own computer (the page explains this).
 *
 *   java -jar minidl.jar site [web-page=web/index.html] [docs-folder=docs]
 */
public final class SiteBuilder {
    private SiteBuilder() {}

    public static void main(String[] args) throws IOException {
        Path page = Paths.get(args.length > 0 ? args[0] : "web/index.html");
        File docs = new File(args.length > 1 ? args[1] : "docs");
        if (!Files.exists(page)) { System.out.println("page not found: " + page + " (run from the project folder)"); return; }
        docs.mkdirs();
        Files.copy(page, new File(docs, "index.html").toPath(), StandardCopyOption.REPLACE_EXISTING);
        Path cam = page.resolveSibling("camera.html");
        if (Files.exists(cam)) Files.copy(cam, new File(docs, "camera.html").toPath(), StandardCopyOption.REPLACE_EXISTING);
        new File(docs, ".nojekyll").createNewFile();
        if (new File(docs, "logs").isDirectory()) Plot.main(new String[]{new File(docs, "logs").getPath(), new File(docs, "mnist-training.svg").getPath()});
        System.out.println("site written to " + docs + ": index.html, camera.html, .nojekyll, mnist-training.svg");
    }
}

// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "AppleMPSBridge",
    platforms: [
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "AppleMPSBridge",
            type: .static,
            targets: ["AppleMPSBridge"]
        )
    ],
    targets: [
        .target(
            name: "AppleMPSBridge",
            path: "Sources/AppleMPSBridge",
            publicHeadersPath: "include"
        )
    ]
)

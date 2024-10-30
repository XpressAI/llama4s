/**
 * *****************************************************************************
 * Global Settings
 * *****************************************************************************
 */

Global / cancelable := true
Global / onChangedBuildSource := ReloadOnSourceChanges

lazy val defaultScalaVersion = "3.5.1"
ThisBuild / scalaVersion := defaultScalaVersion
ThisBuild / organization := "ai.xpress"

inThisBuild(
  List(
    scalaVersion := defaultScalaVersion,
    semanticdbEnabled := true,
    semanticdbVersion := scalafixSemanticdb.revision
  )
)

lazy val LangChain4JVersion = "0.35.0"

libraryDependencies ++= Seq(
  "com.lihaoyi" %% "mainargs" % "0.7.6",
  "com.thesamet.scalapb" %% "scalapb-json4s" % "0.12.0",
  "com.thesamet.scalapb" %% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion % "protobuf",
  "org.json4s" %% "json4s-jackson" % "4.0.7",
  "org.scala-lang" %% "scala3-library" % defaultScalaVersion,
  "org.scalatest" %% "scalatest" % "3.2.19" % Test
)

lazy val root = project
  .in(file("."))
  .settings(
    name := "llmtest",
    ThisBuild / version := "0.1.0",
    ThisBuild / scalacOptions += "-Wunused:all",
    ThisBuild / fork := true,
    ThisBuild / javacOptions ++= Seq(
      "--add-modules=jdk.incubator.vector"
    ),
    ThisBuild / javaOptions ++= Seq(
      "--add-modules=jdk.incubator.vector"
    ),
    Compile / PB.targets := Seq(
      PB.gens.java -> (Compile / sourceManaged).value,
      scalapb.gen() -> (Compile / sourceManaged).value / "scalapb"
    ),
    Compile / PB.protoSources ++= Seq(
      (ThisBuild / baseDirectory).value / "protobuf" / "models"
    ),
    // https://stackoverflow.com/questions/54834125/sbt-assembly-deduplicate-module-info-class
    assembly / assemblyMergeStrategy := {
      case PathList("module-info.class") => MergeStrategy.last
      case path if path.endsWith("/module-info.class") => MergeStrategy.last
      case x =>
        val oldStrategy = (assembly / assemblyMergeStrategy).value
        oldStrategy(x)
    }
  )
